"""Normal-mode analysis of a (partial) Cartesian Hessian.

Frequencies use the same conversion as :class:`ase.vibrations.Vibrations`; imaginary modes
are returned as negative wavenumbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import units

#: sqrt(eV / A^2 / amu) -> eV  (identical to ASE's VibrationsData conversion)
_S = units._hbar * 1e10 / np.sqrt(units._e * units._amu)


@dataclass
class NormalModes:
    frequencies_cm: np.ndarray  #: (m,) signed wavenumbers, negative = imaginary
    modes_mw: np.ndarray  #: (3n, m) orthonormal mass-weighted eigenvectors (columns)
    masses: np.ndarray  #: (n,) masses of the atoms in the Hessian
    n_rigid_removed: int = 0

    @property
    def cartesian_displacements(self) -> np.ndarray:
        """(m, n, 3) Cartesian displacement patterns (mass-weighted vectors / sqrt(m))."""
        inv = np.repeat(1.0 / np.sqrt(self.masses), 3)
        disp = (self.modes_mw * inv[:, None]).T
        return disp.reshape(disp.shape[0], -1, 3)


def eigenvalues_to_cm(eigvals: np.ndarray) -> np.ndarray:
    """Mass-weighted Hessian eigenvalues (eV/A^2/amu) -> signed wavenumbers (cm^-1)."""
    ev = np.asarray(eigvals, dtype=float)
    energies = _S * np.sqrt(np.abs(ev))
    return np.sign(ev) * energies / units.invcm


def rigid_body_basis(
    positions: np.ndarray, masses: np.ndarray, tol: float = 1e-6, linear_tol: float = 1e-3
) -> np.ndarray:
    """Orthonormal basis (3n, k) of mass-weighted translations and rotations (k = 5 or 6).

    Rotations are built about the principal axes of inertia. A rotation whose principal moment is
    below ``linear_tol`` x the largest moment is dropped, so a *numerically* linear molecule
    (e.g. CO2 relaxed by a float32 MLIP to 179.9 deg) keeps 3N-5 vibrations. A plain rank test
    would count that near-zero rotation and project out one component of the degenerate bend.
    """
    pos = np.asarray(positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    n = len(m)
    sqrt_m = np.sqrt(m)
    com = (m[:, None] * pos).sum(0) / m.sum()
    r = pos - com
    vecs = []
    for a in range(3):  # translations
        v = np.zeros((n, 3))
        v[:, a] = sqrt_m
        vecs.append(v.ravel())
    inertia = np.einsum("i,ij,ik->jk", m, r, r)
    inertia = np.trace(inertia) * np.eye(3) - inertia
    moments, axes = np.linalg.eigh(inertia)
    i_max = max(float(moments.max()), 1e-12)
    for k in range(3):  # infinitesimal rotations about the principal axes through the COM
        if moments[k] < linear_tol * i_max:
            continue
        v = np.cross(axes[:, k], r) * sqrt_m[:, None]
        vecs.append(v.ravel())
    mat = np.array(vecs).T
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    rank = int((s > tol * max(s.max(), 1.0)).sum())
    return u[:, :rank]


def normal_modes(
    hessian: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray | None = None,
    project_rigid: bool = False,
) -> NormalModes:
    """Diagonalize the mass-weighted Hessian.

    Args:
        hessian: (3n, 3n) Cartesian Hessian in eV/A^2.
        masses: (n,) atomic masses in amu.
        positions: (n, 3) positions, required when ``project_rigid`` is True.
        project_rigid: remove the 5 (linear) or 6 rigid-body modes exactly by diagonalizing in
            their orthogonal complement. Use only for free molecules with the full Hessian.
    """
    h = np.asarray(hessian, dtype=float)
    m = np.asarray(masses, dtype=float)
    if h.shape != (3 * len(m), 3 * len(m)):
        raise ValueError(f"Hessian shape {h.shape} does not match {len(m)} atoms")
    inv_sqrt = np.repeat(1.0 / np.sqrt(m), 3)
    h_mw = h * inv_sqrt[:, None] * inv_sqrt[None, :]
    h_mw = 0.5 * (h_mw + h_mw.T)

    n_removed = 0
    if project_rigid:
        if positions is None:
            raise ValueError("positions are required to project rigid-body modes")
        rigid = rigid_body_basis(positions, m)
        n_removed = rigid.shape[1]
        # orthonormal complement of the rigid-body subspace
        proj = np.eye(h.shape[0]) - rigid @ rigid.T
        u, s, _ = np.linalg.svd(proj)
        comp = u[:, : h.shape[0] - n_removed]
        evals, evecs = np.linalg.eigh(comp.T @ h_mw @ comp)
        modes = comp @ evecs
    else:
        evals, modes = np.linalg.eigh(h_mw)

    return NormalModes(eigenvalues_to_cm(evals), modes, m, n_removed)
