"""Adsorption-site classification from a relaxed geometry.

Replaces hand labels: the validation runner, the critic node and
``scripts/classify_sites.py`` all call :func:`classify_site`.

Rule: the *anchor* is the adsorbate atom closest to any metal atom. Metal atoms bonded to it
are those within ``bond_tol x (r_cov(anchor) + r_cov(metal))`` (ASE/Cordero covalent radii)
and within ``hop_window`` of the shortest contact. 1 bonded metal atom = ontop, 2 = bridge,
>= 3 = hollow (fcc/hcp from whether a second-layer metal atom sits under the anchor). No bonded
metal atom within the cutoff = physisorbed (shortest contact < 4 A) or desorbed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

ADSORBATE_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl"}

#: requested plan site -> classifier labels that count as "stayed on site"
SITE_EQUIVALENTS = {
    "ontop": {"ontop"},
    "bridge": {"bridge"},
    "fcc": {"fcc", "hollow"},
    "hcp": {"hcp", "hollow"},
    "hollow": {"fcc", "hcp", "hollow"},
}


@dataclass
class SiteInfo:
    site: str  #: ontop | bridge | fcc | hcp | hollow | physisorbed | desorbed
    anchor: str  #: element of the adsorbate atom closest to the metal
    anchor_index: int
    n_metal: int  #: metal atoms bonded to the anchor
    distances: list[float]  #: anchor-metal distances of the bonded atoms (A, sorted)
    min_distance: float  #: shortest adsorbate-metal contact (A)
    cutoff: float  #: bonding cutoff used for the anchor-metal pair (A)

    @property
    def label(self) -> str:
        d = self.distances or [self.min_distance]
        return f"{self.site} {'/'.join(f'{x:.2f}' for x in d)}"

    def as_dict(self) -> dict:
        return {**asdict(self), "label": self.label}


def adsorbate_indices_by_element(atoms: Atoms, elements: set[str] = ADSORBATE_ELEMENTS) -> list[int]:
    """Indices of non-metal atoms (used when only the final geometry is available)."""
    return [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in elements]


def _distances(atoms: Atoms, i: int, js: Sequence[int]) -> np.ndarray:
    return atoms.get_distances(i, list(js), mic=bool(atoms.pbc.any()))


def _hollow_kind(atoms: Atoms, anchor: int, metal: list[int], normal: np.ndarray) -> str:
    """fcc vs hcp: is there a second-layer metal atom (laterally) under the anchor?"""
    pos = atoms.positions
    z = pos[metal] @ normal
    top = z.max()
    second = [m for m, zz in zip(metal, z, strict=True) if top - 3.5 < zz < top - 0.7]
    if not second:
        return "hollow"
    vec = atoms.get_distances(anchor, second, mic=bool(atoms.pbc.any()), vector=True)
    lateral = vec - np.outer(vec @ normal, normal)
    return "hcp" if np.linalg.norm(lateral, axis=1).min() < 0.6 else "fcc"


def classify_site(
    atoms: Atoms,
    adsorbate_indices: Sequence[int] | None = None,
    metal_indices: Sequence[int] | None = None,
    surface_normal: Sequence[float] | None = None,
    bond_tol: float = 1.30,
    hop_window: float = 0.35,
    physisorbed_max: float = 4.0,
) -> SiteInfo:
    ads = list(adsorbate_indices) if adsorbate_indices is not None else adsorbate_indices_by_element(atoms)
    metal = list(metal_indices) if metal_indices is not None else [i for i in range(len(atoms)) if i not in ads]
    if not ads or not metal:
        raise ValueError("need at least one adsorbate and one metal atom")
    normal = np.asarray(surface_normal if surface_normal is not None else (0.0, 0.0, 1.0), float)
    normal = normal / np.linalg.norm(normal)
    dmins = [(_distances(atoms, i, metal).min(), i) for i in ads]
    dmin, anchor = min(dmins)
    d = _distances(atoms, anchor, metal)
    za = atoms.numbers[anchor]
    zm = atoms.numbers[metal[int(np.argmin(d))]]
    cutoff = bond_tol * (covalent_radii[za] + covalent_radii[zm])
    bonded = np.sort(d[(d < cutoff) & (d < dmin + hop_window)])
    n = len(bonded)
    if n == 0:
        site = "physisorbed" if dmin < physisorbed_max else "desorbed"
    elif n == 1:
        site = "ontop"
    elif n == 2:
        site = "bridge"
    else:
        site = _hollow_kind(atoms, anchor, metal, normal)
    return SiteInfo(site, atoms[anchor].symbol, int(anchor), n, [float(x) for x in bonded],
                    float(dmin), float(cutoff))


def site_changed(requested: str, info: SiteInfo) -> bool:
    """True if the relaxed site differs from the requested one (a 'site hop' or desorption)."""
    return info.site not in SITE_EQUIVALENTS.get(requested, {requested})
