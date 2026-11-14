"""Run C structures, band assignment, reference lookup and metal-cluster sizes (no calculators)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

from ir_agent.semis import SYSTEMS, assign_bands, broken_attachments, build
from ir_agent.vibrations.ir import IRResult


@pytest.mark.parametrize("name", SYSTEMS)
@pytest.mark.parametrize("model", ["cluster", "slab"])
def test_structures_are_sane(name, model):
    s = build(name, model)
    a = s.atoms
    expected = {
        ("H/Si(100)-2x1", "cluster"): "H14Si9", ("H/Si(100)-2x1", "slab"): "H6Si12",
        ("NH2+H/Si(100)-2x1", "cluster"): "H15NSi9", ("NH2+H/Si(100)-2x1", "slab"): "H13NSi24",
        ("SiOH/SiO2", "cluster"): "H10O4Si4", ("SiOH/SiO2", "slab"): "H10O4Si4",
    }
    assert s.formula == expected[(name, model)]
    assert bool(a.pbc.any()) == (model == "slab")
    assert broken_attachments(s) == []
    assert set(s.adsorbate_indices) <= set(s.hessian_indices)
    fixed = set(a.constraints[0].index) if a.constraints else set()
    assert not fixed & set(s.hessian_indices)  # partial-Hessian atoms are free
    # every Si is four-fold coordinated except the dimer atoms' bonds to adsorbates
    i, j, d = neighbor_list("ijd", a, 2.6)
    sym = a.get_chemical_symbols()
    for k in [k for k in range(len(a)) if sym[k] == "Si"]:
        partners = [(sym[y], dd) for x, y, dd in zip(i, j, d, strict=True) if x == k]
        bonded = [p for p, dd in partners if (p == "Si" and dd < 2.6) or (p == "H" and dd < 1.7)
                  or (p in "NO" and dd < 1.9)]
        assert len(bonded) == 4, (name, model, k, bonded)
    if "Si(100)" in name:
        dimer = s.hessian_indices[:2]
        assert a.get_distance(*dimer, mic=True) == pytest.approx(2.40, abs=0.01)


def test_si_slab_bottom_is_passivated_and_fixed():
    s = build("H/Si(100)-2x1", "slab")
    a = s.atoms
    z = a.positions[:, 2]
    si = [k for k in range(len(a)) if a[k].symbol == "Si"]
    bottom_si = [k for k in si if z[k] < z[si].min() + 0.3]
    caps = [k for k in range(len(a)) if a[k].symbol == "H" and k not in s.adsorbate_indices]
    assert len(caps) == 2 * len(bottom_si)
    assert set(bottom_si) <= set(a.constraints[0].index)


def _fake_result(atoms, indices, modes_cart, freqs):
    m = atoms.get_masses()[indices]
    mw = np.array([(v * np.sqrt(m)[:, None]).ravel() for v in modes_cart])
    mw /= np.linalg.norm(mw, axis=1)[:, None]
    return IRResult(frequencies_cm=np.array(freqs), intensities_km_mol=None, modes_mw=mw.T,
                    indices=list(indices), route="frequencies-only", force_method="fake", n_displacements=0)


def test_assign_bands_labels_sym_and_asym_si_h():
    s = build("H/Si(100)-2x1", "cluster")
    a = s.atoms
    idx = s.hessian_indices  # [Si_a, Si_b, H_a, H_b]
    (ha, sa), (hb, sb) = s.bonds["Si-H"]
    ua = a.positions[ha] - a.positions[sa]
    ub = a.positions[hb] - a.positions[sb]
    ua, ub = ua / np.linalg.norm(ua), ub / np.linalg.norm(ub)

    def mode(ca, cb):
        v = np.zeros((len(idx), 3))
        v[idx.index(ha)] = ca * ua
        v[idx.index(hb)] = cb * ub
        return v

    lateral = np.zeros((len(idx), 3))
    lateral[idx.index(ha)] = [1.0, 0.0, 0.0]
    res = _fake_result(a, idx, [mode(1, 1), mode(1, -1), lateral], [2110.0, 2095.0, 600.0])
    bands = {b["band"]: b["freq_cm"] for b in assign_bands(res, a, s)}
    assert bands == {"Si-H stretch (sym)": 2110.0, "Si-H stretch (asym)": 2095.0}


def test_bulk_structure_and_nn_scaled_clusters():
    from ir_agent.structures import (
        add_molecule_to_slab,
        build_slab,
        make_cluster_model,
        nearest_neighbour_distance,
    )
    from ir_agent.validation.run import bulk_structure
    from ir_agent.validation.run_c import CLUSTER_RADIUS_NN

    assert bulk_structure("Ru", "0001") == "hcp" and bulk_structure("Cu", "111") == "fcc"
    for metal, st, facet in [("Pt", "fcc", "111"), ("Au", "fcc", "111"), ("Cu", "fcc", "111"), ("Ru", "hcp", "0001")]:
        slab = build_slab(metal, st, facet, (3, 3, 3))
        r = CLUSTER_RADIUS_NN * nearest_neighbour_distance(slab)
        c = make_cluster_model(add_molecule_to_slab(slab, "CO", 2.0, "ontop"), radius=r, n_layers=2)
        assert len(c.slab_indices) == 10, metal


def test_experimental_lookup(tmp_path):
    from ir_agent.validation.experimental import anharmonic_gap, comparison_value

    exp = tmp_path / "exp.csv"
    exp.write_text("system,band,value_cm,technique,coverage,temperature_K,conditions,citation,doi,evidence,verified,compare,note\n"
                   "X/Si,Si-H stretch (sym),2099,MIR-IR,1 ML,300,,A et al.,10.1/x,primary-fulltext,yes,yes,\n"
                   "X/Si,Si-H stretch (sym),2100,HREELS,1 ML,300,,B et al.,10.1/y,secondary,no,no,\n", encoding="utf-8")
    anh = tmp_path / "anh.csv"
    anh.write_text("species,band,nu01_cm,nu02_cm,omega_e_cm,gap_cm,technique,citation,doi,evidence,verified,applies_to,note\n"
                   "Si-H,stretch,2084,4130,2122,38,overtone,C,10.1/z,secondary,no,X/Si:Si-H stretch (sym),\n", encoding="utf-8")
    r = comparison_value("X/Si", "Si-H stretch (sym)", exp)
    assert r["value"] == 2099 and r["verified_bool"] and r["technique"] == "MIR-IR"
    assert comparison_value("X/Si", "N-H stretch (sym)", exp) is None
    g = anharmonic_gap("X/Si", "Si-H stretch (sym)", anh)
    assert g["gap"] == 38 and not g["verified_bool"]
