"""Science pipeline (no LLM): build -> relax -> vibrate/IR -> spectrum -> report.

The LangGraph nodes in :mod:`ir_agent.agent.nodes` and the validation runner both call these
functions, so the deterministic path is exercised by the tests.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.io import write

from ir_agent.agent.plan import SimulationPlan
from ir_agent.calculators.registry import (
    PRESETS,
    make_calculator,
    make_dipole_evaluator,
    make_force_evaluator,
    resolve_dipole_method,
)
from ir_agent.config import AppConfig
from ir_agent.runtime import RunDir, get_logger
from ir_agent.spectrum import Spectrum, broaden
from ir_agent.structures import (
    SurfaceSystem,
    add_molecule_to_slab,
    build_molecule,
    build_slab,
    default_spin_multiplicity,
    make_cluster_model,
    nearest_neighbour_distance,
)
from ir_agent.vibrations.evaluators import ASECalculatorDipoles, extract_fragment
from ir_agent.vibrations.ir import IRResult, compute_ir, compute_ir_ase_infrared

log = get_logger("pipeline")

Kind = Literal["molecule", "cluster", "surface"]


@dataclass
class PreparedSystem:
    atoms: Atoms
    kind: Kind
    adsorbate_indices: list[int]
    charge: int = 0
    spin: int = 1
    surface_normal: np.ndarray | None = None
    surface: SurfaceSystem | None = None
    stages: dict[str, Atoms] = field(default_factory=dict)
    opt_steps: dict[str, int] = field(default_factory=dict)
    converged: dict[str, bool] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)


def _optimizer(name: str):
    from ase import optimize

    return getattr(optimize, name)


def relax(
    atoms: Atoms,
    calc,
    fmax: float,
    cfg: AppConfig,
    label: str,
    run_dir: RunDir | None,
) -> tuple[Atoms, int, bool]:
    """Relax ``atoms`` in place with ``calc``; logs + trajectory go to the run directory.

    Stops early (not converged) when fmax has not improved by 1 % for
    ``optimization.stall_steps`` steps - SCF noise can otherwise keep a tight optimization
    (and its line searches) running for hundreds of force calls.
    """
    atoms.calc = calc
    o = cfg.optimization
    opt_cls = _optimizer(o.optimizer)
    logfile = str(run_dir.path("calc", f"{label}_opt.log")) if run_dir else None
    traj = str(run_dir.path("calc", f"{label}_opt.traj")) if run_dir else None
    opt = opt_cls(atoms, logfile=logfile, trajectory=traj)
    best, stall, converged = np.inf, 0, False
    for converged in opt.irun(fmax=fmax, steps=o.max_steps):
        if converged:
            break
        f = atoms.get_forces()
        cur = float(np.sqrt((f**2).sum(axis=1).max())) if len(f) else 0.0
        if cur < 0.99 * best:
            best, stall = cur, 0
        else:
            stall += 1
        if stall >= o.stall_steps:
            log.warning("%s: fmax stalled at %.4f eV/A for %d steps (target %.3f); stopping",
                        label, cur, stall, fmax)
            break
    steps = int(opt.get_number_of_steps())
    if not converged:
        log.warning("%s: optimizer did not reach fmax=%.3f in %d steps", label, fmax, steps)
    return atoms, steps, bool(converged)


def surface_model_for(plan: SimulationPlan) -> Literal["slab", "cluster"]:
    assert plan.surface is not None
    if plan.surface.model != "auto":
        return plan.surface.model
    return "slab" if PRESETS[plan.calculator].periodic else "cluster"


def prepare_system(plan: SimulationPlan, cfg: AppConfig, run_dir: RunDir | None = None) -> PreparedSystem:
    """Build the initial geometry and pre-optimize (geometry node)."""
    g = cfg.geometry
    t0 = time.perf_counter()
    if plan.surface is None:
        mol = build_molecule(plan.adsorbate, g.molecule_vacuum)
        spin = plan.spin_multiplicity or default_spin_multiplicity(mol, plan.charge)
        init = mol.copy()
        calc = make_calculator(plan.calculator, cfg, atoms=mol, system_kind="molecule",
                               charge=plan.charge, spin=spin, workdir=_wd(run_dir))
        mol, steps, conv = relax(mol, calc, cfg.optimization.fmax_preopt, cfg, "molecule_preopt", run_dir)
        ps = PreparedSystem(mol, "molecule", list(range(len(mol))), plan.charge, spin,
                            stages={"initial": init, "preoptimized": mol.copy()},
                            opt_steps={"preopt": steps}, converged={"preopt": conv})
    else:
        s = plan.surface
        model = surface_model_for(plan)
        slab = build_slab(s.metal, s.structure, s.facet, g.slab_size, g.vacuum, g.fix_bottom_layers)
        stages = {"initial_slab": slab.copy()}
        steps: dict[str, int] = {}
        conv: dict[str, bool] = {}
        if model == "slab":
            calc = make_calculator(plan.calculator, cfg, atoms=slab, system_kind="surface",
                                   workdir=_wd(run_dir))
            slab, steps["bare_slab"], conv["bare_slab"] = relax(slab, calc, cfg.optimization.fmax_preopt,
                                                                cfg, "bare_slab", run_dir)
        stages["relaxed_slab"] = slab.copy()
        system = add_molecule_to_slab(slab, plan.adsorbate, g.adsorbate_height, s.site)
        if model == "cluster":
            radius = (g.cluster_radius if g.cluster_radius_nn is None
                      else g.cluster_radius_nn * nearest_neighbour_distance(slab))
            system = make_cluster_model(system, radius=radius, n_layers=g.cluster_layers)
            log.info("Cluster model: %s (%d metal atoms, non-periodic) for preset '%s'",
                     system.atoms.get_chemical_formula(), len(system.slab_indices), plan.calculator)
        stages["adsorbate_added"] = system.atoms.copy()
        atoms = system.atoms
        kind: Kind = "surface" if model == "slab" else "cluster"
        spin = plan.spin_multiplicity or default_spin_multiplicity(atoms, plan.charge)
        calc = make_calculator(plan.calculator, cfg, atoms=atoms, system_kind=kind,
                               charge=plan.charge, spin=spin, workdir=_wd(run_dir))
        atoms, steps["adsorbate"], conv["adsorbate"] = relax(atoms, calc, cfg.optimization.fmax_preopt,
                                                             cfg, "slab_adsorbate_preopt", run_dir)
        system.atoms = atoms
        stages["preoptimized"] = atoms.copy()
        ps = PreparedSystem(atoms, kind, list(system.adsorbate_indices), plan.charge, spin,
                            surface_normal=system.surface_normal, surface=system, stages=stages,
                            opt_steps=steps, converged=conv)
    ps.timings_s["geometry"] = time.perf_counter() - t0
    if run_dir:
        write(run_dir.path("structures", "preoptimized.xyz"), ps.atoms)
    return ps


def _wd(run_dir: RunDir | None) -> Path | None:
    return run_dir.root / "calc" if run_dir else None


def run_vibrations(
    plan: SimulationPlan, cfg: AppConfig, ps: PreparedSystem, run_dir: RunDir | None = None
) -> IRResult:
    """Tight relaxation + finite-difference IR (calculation node)."""
    v = cfg.vibrations
    atoms = ps.atoms
    t0 = time.perf_counter()
    calc = make_calculator(plan.calculator, cfg, atoms=atoms, system_kind=ps.kind, mode="freq",
                           charge=ps.charge, spin=ps.spin, workdir=_wd(run_dir))
    atoms, steps, converged = relax(atoms, calc, cfg.optimization.fmax_final, cfg, "final", run_dir)
    ps.opt_steps["final"] = steps
    ps.converged["final"] = converged
    ps.stages["final"] = atoms.copy()
    t_opt = time.perf_counter() - t0

    gas = ps.kind == "molecule"
    indices = None if (gas or v.indices == "all") else ps.adsorbate_indices
    normal = None
    if not gas and cfg.ir.surface_selection_rule:
        normal = ps.surface_normal if ps.surface_normal is not None else np.array([0.0, 0.0, 1.0])
    # plan setting wins unless it is 'auto', then the config default applies
    method = plan.dipole_method if plan.dipole_method != "auto" else cfg.ir.dipole_method
    dipole = resolve_dipole_method(method, plan.calculator, cfg)

    surface_like = ps.kind in ("surface", "cluster")

    def fragment_dipoles(dmethod: str):
        """Dipole of the isolated adsorbate fragment (common to every method on surfaces)."""
        region = ps.adsorbate_indices
        frag = extract_fragment(atoms, region)
        return make_dipole_evaluator(dmethod, cfg, charge=0, spin=default_spin_multiplicity(frag, 0),
                                     region=region, workdir=_wd(run_dir))

    if plan.calculator in ("xtb", "xtb-gfn1"):
        # xtb executable: Hessian from xtb's own --hess (finite differences of its single-point
        # --grad gradient split degenerate modes; see XTBBinary.hessian), dipoles by FD.
        t_h = time.perf_counter()
        hess = calc.hessian(atoms)
        t_h = time.perf_counter() - t_h
        label = f"{calc.method_label} (xtb --hess)"
        if surface_like:
            fmethod = method if method not in ("auto", "native") else "xtb"
            de = fragment_dipoles(fmethod)
        else:
            de = ASECalculatorDipoles(calc, name=f"{calc.method_label} (native)")
        res = compute_ir(atoms, None, de, indices=indices, delta=v.delta, nfree=v.nfree,
                         project_rigid=gas and v.project_rigid_modes, surface_normal=normal,
                         image_factor=cfg.ir.image_dipole_factor, hessian=hess, hessian_label=label)
        res.route = "xtb-hessian"
        res.timings_s["xtb_hessian"] = t_h
        if surface_like:
            alt = compute_ir(atoms, None, ASECalculatorDipoles(calc, name="cluster"), indices=indices,
                             delta=v.delta, nfree=v.nfree, surface_normal=normal,
                             image_factor=cfg.ir.image_dipole_factor, hessian=hess)
            res.alt_intensities_km_mol[f"whole-{ps.kind} dipole ({calc.method_label})"] = alt.intensities_km_mol
            res.notes.append(
                "Reported intensities use the adsorbate-fragment dipole (same model as the MLIP rows); "
                f"the whole-{ps.kind} {calc.method_label} dipole (includes metal polarization of a finite "
                "cluster) is stored separately and is not comparable."
            )
    elif dipole == plan.calculator:  # other native dipoles (tblite, GPAW, VASP, dxtb) -> ASE Infrared
        with contextlib.ExitStack() as stack:
            wd = (run_dir.root / "calc" / "ir") if run_dir else Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="ir_agent_"))
            )
            res = compute_ir_ase_infrared(
                atoms, indices, wd, delta=v.delta, nfree=v.nfree,
                project_rigid=gas and v.project_rigid_modes, surface_normal=normal,
                image_factor=cfg.ir.image_dipole_factor,
            )
    else:
        fe = make_force_evaluator(plan.calculator, cfg, atoms=atoms, system_kind=ps.kind,
                                  charge=ps.charge, spin=ps.spin, workdir=_wd(run_dir))
        de = None
        if dipole is not None:
            region = None
            charge, spin = ps.charge, ps.spin
            if ps.kind in ("surface", "cluster"):  # dipole of the isolated adsorbate fragment
                region = ps.adsorbate_indices
                frag = extract_fragment(atoms, region)
                charge, spin = 0, default_spin_multiplicity(frag, 0)
            de = make_dipole_evaluator(dipole, cfg, charge=charge, spin=spin, region=region,
                                       workdir=_wd(run_dir))
        res = compute_ir(atoms, fe, de, indices=indices, delta=v.delta, nfree=v.nfree,
                         project_rigid=gas and v.project_rigid_modes, surface_normal=normal,
                         image_factor=cfg.ir.image_dipole_factor)
        if dipole is not None and ps.kind in ("surface", "cluster"):
            res.notes.append(
                f"Hybrid intensities: Hessian from the force method on the full {ps.kind}, dipoles from "
                f"{dipole} on the isolated adsorbate at each displaced geometry (no metal "
                "screening/charge transfer in the dipole; MSSR applied)."
            )
    if dipole == "dxtb":
        from ir_agent.calculators.dxtb_calc import libcint_available

        if not libcint_available():
            res.dipole_method = (res.dipole_method or "dxtb") + " [Mulliken-charge dipole]"
            res.notes.append("dxtb without tad-libcint: dipole = sum q_i r_i (charge flux kept, "
                             "atomic polarization missing); intensities are approximate.")
    if not converged:
        res.notes.append("Final optimization did not converge; frequencies may be unreliable.")
    res.timings_s["final_opt"] = t_opt
    if run_dir:
        write(run_dir.path("structures", "final.xyz"), atoms)
    return res


def make_spectrum(res: IRResult, cfg: AppConfig) -> Spectrum:
    s = cfg.spectrum
    return broaden(res.frequencies_cm, res.intensities_km_mol, s.fwhm_cm, s.lineshape,
                   s.x_min_cm, s.x_max_cm, s.n_points, s.scale_factor, cfg.vibrations.min_frequency_cm)


def run_pipeline(plan: SimulationPlan, cfg: AppConfig, run_dir: RunDir | None = None) -> dict[str, Any]:
    """Run everything without LangGraph/LLM (used by validation and tests)."""
    from ir_agent.analysis import render_report

    ps = prepare_system(plan, cfg, run_dir)
    res = run_vibrations(plan, cfg, ps, run_dir)
    spec = make_spectrum(res, cfg)
    report, summary = render_report(
        plan_desc=plan.description, calculator=plan.calculator, result=res, atoms=ps.stages["final"],
        adsorbate=plan.adsorbate, metal=plan.surface.metal if plan.surface else None,
        min_cm=cfg.vibrations.min_frequency_cm, scale_factor=cfg.spectrum.scale_factor,
    )
    return {"prepared": ps, "result": res, "spectrum": spec, "report": report, "summary": summary}
