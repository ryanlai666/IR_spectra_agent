from __future__ import annotations

import numpy as np
import pytest

from ir_agent.structures import (
    add_molecule_to_slab,
    build_molecule,
    build_slab,
    default_spin_multiplicity,
    make_cluster_model,
)


@pytest.mark.parametrize(
    "structure,facet,n_atoms",
    [("fcc", "111", 27), ("fcc", "100", 27), ("bcc", "110", 27), ("hcp", "0001", 27)],
)
def test_build_slab(structure, facet, n_atoms):
    metal = {"fcc": "Pt", "bcc": "Fe", "hcp": "Ru"}[structure]
    slab = build_slab(metal, structure, facet, (3, 3, 3), vacuum=10.0, fix_bottom_layers=2)
    assert len(slab) == n_atoms
    assert slab.pbc.all()
    fixed = slab.constraints[0].get_indices()
    assert len(fixed) == 18  # bottom 2 of 3 layers
    z = slab.positions[:, 2]
    assert z[fixed].max() < z.max()


def test_build_slab_rejects_bad_input():
    with pytest.raises(ValueError, match="Unsupported"):
        build_slab("Pt", "fcc", "0001")
    with pytest.raises(ValueError, match="element"):
        build_slab("Xx", "fcc", "111")


@pytest.mark.parametrize("name,binder", [("CO", "C"), ("NH3", "N"), ("H2O", "O")])
def test_adsorbate_binds_through_expected_atom(name, binder):
    slab = build_slab("Pt", "fcc", "111", (2, 2, 3))
    sys_ = add_molecule_to_slab(slab, name, height=2.0)
    atoms = sys_.atoms
    ads = sys_.adsorbate_indices
    assert len(ads) == len(build_molecule(name))
    assert atoms[sys_.binding_index].symbol == binder
    # binding atom is the lowest adsorbate atom and sits 2.0 A above the top layer
    z = atoms.positions[:, 2]
    assert z[sys_.binding_index] == pytest.approx(z[ads].min())
    assert z[sys_.binding_index] - z[sys_.slab_indices].max() == pytest.approx(2.0, abs=1e-6)


def test_cluster_model_pt111_ontop():
    slab = build_slab("Pt", "fcc", "111", (3, 3, 3))
    system = add_molecule_to_slab(slab, "CO", height=1.9)
    cl = make_cluster_model(system, radius=3.0, n_layers=2)
    atoms = cl.atoms
    assert not atoms.pbc.any()
    assert atoms.get_chemical_symbols().count("Pt") == 10  # 7 top + 3 second layer
    assert len(cl.adsorbate_indices) == 2
    assert atoms[cl.binding_index].symbol == "C"
    # metal fixed, adsorbate free
    assert set(atoms.constraints[0].get_indices()) == set(range(10))
    # C sits above the central Pt atom
    pt = atoms.positions[:10]
    top = pt[pt[:, 2] > pt[:, 2].max() - 0.5]
    c = atoms.positions[cl.binding_index]
    d_xy = np.linalg.norm(top[:, :2] - c[:2], axis=1)
    assert d_xy.min() < 1e-6


def test_spin_multiplicity():
    assert default_spin_multiplicity(build_molecule("H2O")) == 1
    assert default_spin_multiplicity(build_molecule("NO")) == 2
    assert default_spin_multiplicity(build_molecule("H2O"), charge=1) == 2


def test_unknown_molecule():
    with pytest.raises(ValueError, match="Unknown molecule"):
        build_molecule("NotAMolecule")
