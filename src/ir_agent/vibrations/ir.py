"""IR spectrum drivers.

Two routes produce an :class:`IRResult`:

* :func:`compute_ir` - the *hybrid* finite-difference route. The Hessian comes from any force
  evaluator (EMT, UMA batched on GPU, xTB, ...) and the dipole derivatives from a (possibly
  different) dipole-capable method evaluated at the *same* displaced geometries.
* :func:`compute_ir_ase_infrared` - ASE's :class:`ase.vibrations.Infrared`, kept for
  calculators that provide dipoles natively (GPAW, VASP, xTB, tblite).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ir_agent.runtime import get_logger
from ir_agent.vibrations.evaluators import DipoleEvaluator, ForceEvaluator
from ir_agent.vibrations.finite_diff import (
    assemble_dipole_derivatives,
    assemble_hessian,
    make_displacements,
)
from ir_agent.vibrations.intensities import dipole_derivatives_wrt_modes, ir_intensities
from ir_agent.vibrations.normal_modes import normal_modes

log = get_logger("ir")


@dataclass
class IRResult:
    frequencies_cm: np.ndarray  #: (m,) signed harmonic wavenumbers (negative = imaginary)
    intensities_km_mol: np.ndarray | None  #: (m,) intensities used for the spectrum
    intensities_isotropic_km_mol: np.ndarray | None = None  #: (m,) full |dmu/dQ|^2
    dmu_dq: np.ndarray | None = None  #: (m, 3) D/(A amu^1/2)
    modes_mw: np.ndarray | None = None  #: (3n, m) mass-weighted eigenvectors
    indices: list[int] = field(default_factory=list)
    route: str = "hybrid"  #: hybrid | ase-infrared | frequencies-only
    force_method: str = ""
    dipole_method: str | None = None
    surface_selection_rule: bool = False
    n_displacements: int = 0
    timings_s: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: alternative intensity sets for the same modes (e.g. whole-cluster dipole), km/mol
    alt_intensities_km_mol: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_intensities(self) -> bool:
        return self.intensities_km_mol is not None

    def real_modes(self, min_cm: float = 50.0) -> np.ndarray:
        """Boolean mask of real modes above ``min_cm``."""
        return self.frequencies_cm > min_cm

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("modes_mw", None)
        return d


def compute_ir(
    atoms: Atoms,
    force_evaluator: ForceEvaluator | None,
    dipole_evaluator: DipoleEvaluator | None = None,
    indices: Sequence[int] | None = None,
    delta: float = 0.01,
    nfree: int = 2,
    project_rigid: bool = False,
    surface_normal: np.ndarray | None = None,
    image_factor: float = 1.0,
    hessian: np.ndarray | None = None,
    hessian_label: str | None = None,
) -> IRResult:
    """Hybrid finite-difference IR calculation.

    Args:
        atoms: relaxed structure.
        force_evaluator: maps displaced images -> forces (n_images, n_atoms, 3).
        dipole_evaluator: maps the same images -> dipoles (n_images, 3); None = frequencies only.
        indices: atoms to displace (default: all). Partial Hessian for a subset.
        project_rigid: project out translations/rotations (free molecule, all atoms).
        surface_normal: if set, apply the metal-surface selection rule along this normal.
        hessian: precomputed Cartesian Hessian (eV/A^2), either full (3N x 3N) or already
            restricted to ``indices``; when given, no forces are evaluated (e.g. xtb's own
            ``--hess``) and the displaced geometries are used for dipoles only.
    """
    idx = list(range(len(atoms))) if indices is None else [int(i) for i in indices]
    if project_rigid and len(idx) != len(atoms):
        raise ValueError("project_rigid requires displacing all atoms")
    disps, images = make_displacements(atoms, idx, delta, nfree)
    timings: dict[str, float] = {}

    if hessian is not None:
        h = np.asarray(hessian, dtype=float)
        if h.shape == (3 * len(atoms), 3 * len(atoms)) and len(idx) != len(atoms):
            dof = np.array([3 * i + k for i in idx for k in range(3)])
            h = h[np.ix_(dof, dof)]  # frozen-environment partial Hessian = sub-block
        if h.shape != (3 * len(idx), 3 * len(idx)):
            raise ValueError(f"hessian shape {h.shape} does not match {len(idx)} displaced atoms")
        hess = 0.5 * (h + h.T)
        force_name = hessian_label or "precomputed Hessian"
    else:
        if force_evaluator is None:
            raise ValueError("need a force_evaluator or a precomputed hessian")
        t0 = time.perf_counter()
        forces = np.asarray(force_evaluator(images))
        timings["forces"] = time.perf_counter() - t0
        log.info(
            "Hessian: %d displaced geometries with %s in %.2f s",
            len(images), force_evaluator.name, timings["forces"],
        )
        hess = assemble_hessian(forces, disps, idx, delta, nfree)
        force_name = force_evaluator.name
    masses = atoms.get_masses()[idx]
    nm = normal_modes(hess, masses, atoms.get_positions()[idx], project_rigid=project_rigid)

    result = IRResult(
        frequencies_cm=nm.frequencies_cm,
        intensities_km_mol=None,
        modes_mw=nm.modes_mw,
        indices=idx,
        route="frequencies-only",
        force_method=force_name,
        n_displacements=len(images),
        timings_s=timings,
    )
    if dipole_evaluator is None:
        result.notes.append("No dipole method available: intensities not computed.")
        return result

    t0 = time.perf_counter()
    dipoles = np.asarray(dipole_evaluator(images))
    timings["dipoles"] = time.perf_counter() - t0
    log.info("Dipoles: %d geometries with %s in %.2f s", len(images), dipole_evaluator.name, timings["dipoles"])

    dmu_dx = assemble_dipole_derivatives(dipoles, disps, idx, delta, nfree)
    dmu_dq = dipole_derivatives_wrt_modes(dmu_dx, nm.modes_mw, masses)
    iso = ir_intensities(dmu_dq)
    used = iso if surface_normal is None else ir_intensities(dmu_dq, surface_normal, image_factor)
    result.intensities_km_mol = used
    result.intensities_isotropic_km_mol = iso
    result.dmu_dq = dmu_dq
    result.route = "hybrid"
    result.dipole_method = dipole_evaluator.name
    result.surface_selection_rule = surface_normal is not None
    if surface_normal is not None:
        result.notes.append(
            "Metal-surface selection rule applied: intensities use only the surface-normal "
            "component of dmu/dQ."
        )
    return result


def compute_ir_ase_infrared(
    atoms: Atoms,
    indices: Sequence[int] | None,
    workdir: Path,
    delta: float = 0.01,
    nfree: int = 2,
    project_rigid: bool = False,
    surface_normal: np.ndarray | None = None,
    image_factor: float = 1.0,
) -> IRResult:
    """ASE :class:`~ase.vibrations.Infrared` route for calculators with native dipoles.

    ASE runs and caches the displaced calculations (``workdir/ir.*``) and assembles the
    Hessian (``Infrared.H``). We then (optionally) project out rigid-body modes -- which ASE
    does not do, so residual gradients leave spurious "rotations" at a few hundred cm^-1 for
    free molecules -- and compute intensities from the cached displacement dipoles, with the
    metal-surface selection rule along any ``surface_normal``. Without projection/MSSR the
    result equals ASE's own ``Infrared.intensities`` (checked in the tests).
    """
    from ase.vibrations import Infrared

    from ir_agent.vibrations.finite_diff import stencil

    idx = list(range(len(atoms))) if indices is None else [int(i) for i in indices]
    if project_rigid and len(idx) != len(atoms):
        raise ValueError("project_rigid requires displacing all atoms")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ir = Infrared(atoms, indices=idx, delta=delta, nfree=nfree, name=str(workdir / "ir"))
    ir.run()
    ir.read()
    hess = np.asarray(ir.H, dtype=float)  # eV/A^2, symmetrized by ASE
    weights, denom = stencil(nfree)
    dmu_dx = np.zeros((3 * len(idx), 3))
    for r, (a, i) in enumerate(ir._iter_ai()):
        acc = np.zeros(3)
        for step, w in weights.items():
            acc += w * np.asarray(ir._disp(a, i, step).dipole(), dtype=float)
        dmu_dx[r] = acc / (denom * delta)
    timings = {"ase_infrared": time.perf_counter() - t0}

    masses = atoms.get_masses()[idx]
    nm = normal_modes(hess, masses, atoms.get_positions()[idx], project_rigid=project_rigid)
    dmu_dq = dipole_derivatives_wrt_modes(dmu_dx, nm.modes_mw, masses)
    iso = ir_intensities(dmu_dq)
    used = iso if surface_normal is None else ir_intensities(dmu_dq, surface_normal, image_factor)
    calc_name = getattr(atoms.calc, "method_label", None) or getattr(atoms.calc, "name", type(atoms.calc).__name__)
    res = IRResult(
        frequencies_cm=nm.frequencies_cm,
        intensities_km_mol=used,
        intensities_isotropic_km_mol=iso,
        dmu_dq=dmu_dq,
        modes_mw=nm.modes_mw,
        indices=idx,
        route="ase-infrared",
        force_method=calc_name,
        dipole_method=f"{calc_name} (native)",
        surface_selection_rule=surface_normal is not None,
        n_displacements=len(idx) * 3 * nfree,
        timings_s=timings,
    )
    if surface_normal is not None:
        res.notes.append("Metal-surface selection rule applied (surface-normal dmu/dQ only).")
    return res
