"""Finite-difference displacements, Hessians and dipole derivatives.

All displaced geometries are generated up front so that a batched evaluator (e.g. UMA on a
GPU) can process them in a few forward passes, and so that the *same* displaced geometries
can be fed to a different, dipole-capable method (hybrid IR scheme).

Units follow ASE: positions in Angstrom, forces in eV/Angstrom, dipoles in e*Angstrom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

# Central-difference stencils: displacement multiple -> weight, divided by (denominator * delta)
_STENCILS: dict[int, tuple[dict[int, float], float]] = {
    2: ({+1: 1.0, -1: -1.0}, 2.0),
    4: ({+2: -1.0, +1: 8.0, -1: -8.0, -2: 1.0}, 12.0),
}


@dataclass(frozen=True)
class Displacement:
    atom: int  #: index of the displaced atom
    axis: int  #: 0, 1, 2 for x, y, z
    step: int  #: signed multiple of ``delta``


def stencil(nfree: int) -> tuple[dict[int, float], float]:
    if nfree not in _STENCILS:
        raise ValueError(f"nfree must be one of {sorted(_STENCILS)}, got {nfree}")
    return _STENCILS[nfree]


def make_displacements(
    atoms: Atoms, indices: Sequence[int], delta: float = 0.01, nfree: int = 2
) -> tuple[list[Displacement], list[Atoms]]:
    """Return the displacement list and the displaced copies of ``atoms`` (no calculator)."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    weights, _ = stencil(nfree)
    steps = sorted(weights, key=lambda s: (abs(s), s))
    disps: list[Displacement] = []
    images: list[Atoms] = []
    base = atoms.copy()
    base.calc = None
    for i in indices:
        for axis in range(3):
            for step in steps:
                img = base.copy()
                pos = img.get_positions()
                pos[i, axis] += step * delta
                img.set_positions(pos, apply_constraint=False)
                disps.append(Displacement(int(i), axis, step))
                images.append(img)
    return disps, images


def _derivative_rows(
    values: np.ndarray, disps: Sequence[Displacement], indices: Sequence[int], delta: float, nfree: int
) -> np.ndarray:
    """d(values)/dx for each displaced coordinate -> array (3*len(indices), *values.shape[1:])."""
    weights, denom = stencil(nfree)
    lookup = {(d.atom, d.axis, d.step): k for k, d in enumerate(disps)}
    rows = []
    for i in indices:
        for axis in range(3):
            acc = 0.0
            for step, w in weights.items():
                acc = acc + w * values[lookup[(int(i), axis, step)]]
            rows.append(acc / (denom * delta))
    return np.asarray(rows)


def assemble_hessian(
    forces: np.ndarray,
    disps: Sequence[Displacement],
    indices: Sequence[int],
    delta: float,
    nfree: int = 2,
    symmetrize: bool = True,
) -> np.ndarray:
    """Partial Hessian (3n x 3n, eV/A^2) over ``indices`` from forces at displaced geometries.

    ``forces`` has shape (n_images, n_atoms, 3) in eV/A. H_ij = -dF_j/dx_i.
    """
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 3 or forces.shape[0] != len(disps):
        raise ValueError(f"forces must be (n_images={len(disps)}, n_atoms, 3); got {forces.shape}")
    idx = np.asarray(indices, dtype=int)
    sub = forces[:, idx, :].reshape(len(disps), -1)  # (n_images, 3n)
    hess = -_derivative_rows(sub, disps, indices, delta, nfree)
    if symmetrize:
        hess = 0.5 * (hess + hess.T)
    return hess


def assemble_dipole_derivatives(
    dipoles: np.ndarray,
    disps: Sequence[Displacement],
    indices: Sequence[int],
    delta: float,
    nfree: int = 2,
) -> np.ndarray:
    """Dipole derivatives d(mu_beta)/d(x_i) as (3n, 3) in e (i.e. e*A/A)."""
    dipoles = np.asarray(dipoles, dtype=float)
    if dipoles.shape != (len(disps), 3):
        raise ValueError(f"dipoles must be (n_images={len(disps)}, 3); got {dipoles.shape}")
    return _derivative_rows(dipoles, disps, indices, delta, nfree)
