"""Structure builders: gas-phase molecules, metal slabs, adsorbate placement, cluster models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.build import (
    add_adsorbate,
    bcc100,
    bcc110,
    bcc111,
    fcc100,
    fcc110,
    fcc111,
    fcc211,
    hcp0001,
    hcp10m10,
    molecule,
)
from ase.collections import g2
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, chemical_symbols

SLAB_BUILDERS = {
    "fcc": {"100": fcc100, "110": fcc110, "111": fcc111, "211": fcc211},
    "bcc": {"100": bcc100, "110": bcc110, "111": bcc111},
    "hcp": {"0001": hcp0001, "10m10": hcp10m10},
}

#: Atom through which common molecules bind to a metal surface.
DEFAULT_BINDING_ATOM = {
    "CO": "C",
    "NH3": "N",
    "H2O": "O",
    "N2": "N",
    "NO": "N",
    "CO2": "O",
    "CH4": "C",
    "H2": "H",
    "O2": "O",
    "HCN": "N",
    "CH3OH": "O",
}


@dataclass
class SurfaceSystem:
    """Slab + adsorbate with bookkeeping the vibrational analysis needs."""

    atoms: Atoms
    slab_indices: list[int]
    adsorbate_indices: list[int]
    binding_index: int
    surface_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


def available_molecules() -> list[str]:
    return sorted(g2.names)


def validate_molecule_name(name: str) -> str:
    """Return ``name`` if ASE can build it (g2 collection or extra ASE molecules)."""
    try:
        molecule(name)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Unknown molecule '{name}'. Use an ASE g2 name such as H2O, CO, NH3, CO2, CH4."
        ) from exc
    return name


def is_element(symbol: str) -> bool:
    return symbol in atomic_numbers and symbol != "X"


def build_molecule(name: str, vacuum: float = 8.0) -> Atoms:
    """Isolated molecule in a box (non-periodic)."""
    validate_molecule_name(name)
    mol = molecule(name, vacuum=vacuum)
    mol.pbc = False
    return mol


def default_spin_multiplicity(atoms: Atoms, charge: int = 0) -> int:
    """Lowest multiplicity compatible with the electron count (1 for even, 2 for odd)."""
    n_el = int(atoms.get_atomic_numbers().sum()) - int(charge)
    return 1 if n_el % 2 == 0 else 2


def build_slab(
    metal: str,
    structure: str = "fcc",
    facet: str = "111",
    size: Sequence[int] = (3, 3, 3),
    vacuum: float = 10.0,
    fix_bottom_layers: int = 2,
) -> Atoms:
    """Periodic metal slab (surface normal along +z) with the bottom layers fixed."""
    if not is_element(metal):
        raise ValueError(f"'{metal}' is not a chemical element symbol")
    builder = SLAB_BUILDERS.get(structure, {}).get(facet)
    if builder is None:
        supported = {k: sorted(v) for k, v in SLAB_BUILDERS.items()}
        raise ValueError(f"Unsupported surface {structure}({facet}). Supported: {supported}")
    slab = builder(metal, size=tuple(size), vacuum=vacuum)
    slab.pbc = (True, True, True)
    _fix_bottom_layers(slab, fix_bottom_layers)
    return slab


def _fix_bottom_layers(slab: Atoms, n_fixed: int) -> None:
    if n_fixed <= 0:
        return
    from ase.geometry import get_layers

    layers, _ = get_layers(slab, (0, 0, 1), tolerance=0.3)
    fixed = np.where(layers < n_fixed)[0]
    slab.set_constraint(FixAtoms(indices=fixed))


def orient_for_adsorption(mol: Atoms, binding_index: int) -> Atoms:
    """Rotate so the binding atom points down (-z) and the rest of the molecule points up."""
    mol = mol.copy()
    com = mol.get_center_of_mass()
    v = com - mol.positions[binding_index]
    if np.linalg.norm(v) > 1e-3:
        mol.rotate(v, "z", center=mol.positions[binding_index])
    return mol


def add_molecule_to_slab(
    slab: Atoms,
    name: str,
    height: float = 2.0,
    site: str = "ontop",
    binding_atom: str | None = None,
) -> SurfaceSystem:
    """Place molecule ``name`` on ``slab`` through its binding atom."""
    mol = build_molecule(name, vacuum=0.0)
    mol.cell = np.zeros((3, 3))
    symbol = binding_atom or DEFAULT_BINDING_ATOM.get(name)
    syms = mol.get_chemical_symbols()
    if symbol is not None and symbol in syms:
        b = syms.index(symbol)
    else:
        b = int(np.argmax([atomic_numbers[s] for s in syms]))  # heaviest atom
    mol = orient_for_adsorption(mol, b)
    system = slab.copy()
    n_slab = len(system)
    try:
        add_adsorbate(system, mol, height=height, position=site, mol_index=b)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Site '{site}' is not defined for this slab") from exc
    ads = list(range(n_slab, len(system)))
    return SurfaceSystem(
        atoms=system,
        slab_indices=list(range(n_slab)),
        adsorbate_indices=ads,
        binding_index=n_slab + b,
    )


def nearest_neighbour_distance(atoms: Atoms, cutoff: float = 4.0) -> float:
    """Shortest interatomic distance (minimum image), e.g. the metal-metal distance of a slab."""
    from ase.neighborlist import neighbor_list

    d = neighbor_list("d", atoms, cutoff)
    if not len(d):
        raise ValueError(f"no neighbours within {cutoff} A")
    return float(d.min())


def make_cluster_model(
    system: SurfaceSystem,
    radius: float = 4.0,
    n_layers: int = 2,
    vacuum: float = 8.0,
) -> SurfaceSystem:
    """Cut a finite metal cluster around the binding site (for non-periodic methods like xTB).

    Keeps metal atoms in the top ``n_layers`` whose lateral distance (minimum image) to the
    binding atom is <= ``radius``. Metal atoms are fixed; the adsorbate is free. The result is
    non-periodic and keeps the slab orientation (surface normal +z).
    """
    from ase.geometry import find_mic, get_layers

    atoms = system.atoms
    slab = atoms[system.slab_indices]
    layers, _ = get_layers(slab, (0, 0, 1), tolerance=0.3)
    top = layers.max()
    site = atoms.positions[system.binding_index]
    keep = []
    shifted = []
    for k, i in enumerate(system.slab_indices):
        if layers[k] <= top - n_layers:
            continue
        d = atoms.positions[i] - site
        d_xy = d.copy()
        d_xy[2] = 0.0
        mic, _ = find_mic(d_xy[None, :], atoms.cell, pbc=[True, True, False])
        if np.linalg.norm(mic[0]) <= radius + 1e-6:
            keep.append(i)
            shifted.append(site + mic[0] + np.array([0.0, 0.0, d[2]]))
    if not keep:
        raise ValueError("Cluster radius too small: no metal atoms selected")
    metal = Atoms(
        [atoms[i].symbol for i in keep], positions=np.array(shifted), pbc=False
    )
    ads = atoms[system.adsorbate_indices].copy()
    ads.set_constraint()
    cluster = metal + Atoms(ads.get_chemical_symbols(), positions=ads.positions, pbc=False)
    cluster.set_cell(np.zeros((3, 3)))
    cluster.center(vacuum=vacuum)
    cluster.pbc = False
    n_m = len(metal)
    cluster.set_constraint(FixAtoms(indices=list(range(n_m))))
    b_local = system.adsorbate_indices.index(system.binding_index)
    return SurfaceSystem(
        atoms=cluster,
        slab_indices=list(range(n_m)),
        adsorbate_indices=list(range(n_m, len(cluster))),
        binding_index=n_m + b_local,
    )


def element_set(atoms: Atoms) -> set[str]:
    return set(atoms.get_chemical_symbols())


def check_symbols(symbols: Sequence[str]) -> None:
    bad = [s for s in symbols if s not in chemical_symbols]
    if bad:
        raise ValueError(f"Unknown element symbols: {bad}")
