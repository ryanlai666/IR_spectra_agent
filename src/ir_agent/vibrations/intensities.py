"""IR intensities from dipole derivatives (double-harmonic approximation).

    I_k = (N_A * pi / 3c^2) * |d mu / d Q_k|^2

With dmu/dQ in Debye/(Angstrom*sqrt(amu)), 1 (D/A)^2/amu = 42.2561 km/mol, the conversion
used by ASE's ``Infrared.summary(intensity_unit='km/mol')`` (42.255) and most QC codes.

Metal-surface selection rule (MSSR): for an adsorbate on a metal, the field parallel to the
surface is screened and the image dipole doubles the perpendicular component, so only the
projection of dmu/dQ on the surface normal is IR active (RAIRS). We therefore use
I_k ~ factor * (dmu/dQ_k . n)^2 with a configurable image factor (default 1, i.e. relative
intensities only; 4 adds the classical image-dipole enhancement).
"""

from __future__ import annotations

import numpy as np
from ase import units

#: 1 e*A/A expressed in Debye/A (1 e*A = 4.80320 D)
DEBYE_PER_E_ANG = 1.0 / units.Debye
#: 1 (D/A)^2/amu in km/mol
KMMOL_PER_D2_A2_AMU = 42.2561


def dipole_derivatives_wrt_modes(
    dmu_dx: np.ndarray, modes_mw: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Project Cartesian dipole derivatives onto normal modes.

    Args:
        dmu_dx: (3n, 3) d(mu_beta)/d(x_i) in e (e*A/A).
        modes_mw: (3n, m) orthonormal mass-weighted normal modes.
        masses: (n,) masses (amu).

    Returns:
        (m, 3) dmu/dQ in D/(A*sqrt(amu)).
    """
    inv = np.repeat(1.0 / np.sqrt(np.asarray(masses, dtype=float)), 3)
    cart = modes_mw * inv[:, None]  # (3n, m): dx/dQ
    return (cart.T @ np.asarray(dmu_dx, dtype=float)) * DEBYE_PER_E_ANG


def ir_intensities(
    dmu_dq: np.ndarray,
    surface_normal: np.ndarray | None = None,
    image_factor: float = 1.0,
) -> np.ndarray:
    """IR intensities in km/mol.

    If ``surface_normal`` is given, apply the metal-surface selection rule (only the normal
    component of dmu/dQ contributes, scaled by ``image_factor``).
    """
    d = np.asarray(dmu_dq, dtype=float)
    if surface_normal is None:
        sq = (d**2).sum(axis=1)
    else:
        n = np.asarray(surface_normal, dtype=float)
        n = n / np.linalg.norm(n)
        sq = image_factor * (d @ n) ** 2
    return sq * KMMOL_PER_D2_A2_AMU


def surface_normal_from_cell(cell: np.ndarray) -> np.ndarray:
    """Unit vector a x b of a slab cell (the surface normal for ASE-built slabs)."""
    c = np.asarray(cell, dtype=float)
    n = np.cross(c[0], c[1])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    n = n / norm
    return n if n[2] >= 0 else -n
