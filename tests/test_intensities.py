"""Finite differences, normal modes and IR intensity math on toy systems with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from ase import units

from ir_agent.vibrations.evaluators import (
    ASECalculatorDipoles,
    ASECalculatorForces,
    FixedChargeDipoles,
)
from ir_agent.vibrations.finite_diff import assemble_hessian, make_displacements
from ir_agent.vibrations.intensities import (
    DEBYE_PER_E_ANG,
    KMMOL_PER_D2_A2_AMU,
    ir_intensities,
    surface_normal_from_cell,
)
from ir_agent.vibrations.ir import compute_ir, compute_ir_ase_infrared
from ir_agent.vibrations.normal_modes import eigenvalues_to_cm, normal_modes, rigid_body_basis


def analytic_diatomic(atoms, k, q):
    m1, m2 = atoms.get_masses()
    mu = m1 * m2 / (m1 + m2)
    nu_cm = eigenvalues_to_cm(np.array([k / mu]))[0]
    inten = (q * DEBYE_PER_E_ANG) ** 2 / mu * KMMOL_PER_D2_A2_AMU
    return nu_cm, inten


@pytest.mark.parametrize("nfree", [2, 4])
def test_diatomic_frequency_and_intensity_match_analytic(diatomic, nfree):
    atoms, calc, p = diatomic
    res = compute_ir(atoms, ASECalculatorForces(calc), ASECalculatorDipoles(calc),
                     delta=0.005, nfree=nfree, project_rigid=True)
    assert len(res.frequencies_cm) == 1  # 3N - 5 for a linear molecule
    nu, inten = analytic_diatomic(atoms, p["k"], p["q"])
    assert res.frequencies_cm[0] == pytest.approx(nu, rel=1e-4)
    assert res.intensities_km_mol[0] == pytest.approx(inten, rel=1e-4)


def test_fixed_charge_dipoles_equal_calculator_dipoles(diatomic):
    atoms, calc, p = diatomic
    a = compute_ir(atoms, ASECalculatorForces(calc), ASECalculatorDipoles(calc), project_rigid=True)
    b = compute_ir(atoms, ASECalculatorForces(calc), FixedChargeDipoles([p["q"], -p["q"]]), project_rigid=True)
    np.testing.assert_allclose(a.intensities_km_mol, b.intensities_km_mol, rtol=1e-10)


def test_hybrid_matches_ase_infrared(triatomic, tmp_path):
    """Our hybrid route with the same calculator for forces and dipoles == ase.vibrations.Infrared."""
    atoms, calc = triatomic
    ours = compute_ir(atoms, ASECalculatorForces(calc), ASECalculatorDipoles(calc), delta=0.01)
    a2 = atoms.copy()
    a2.calc = calc
    ref = compute_ir_ase_infrared(a2, None, tmp_path / "ir", delta=0.01)
    keep_o = ours.frequencies_cm > 100
    keep_r = ref.frequencies_cm > 100
    np.testing.assert_allclose(np.sort(ours.frequencies_cm[keep_o]), np.sort(ref.frequencies_cm[keep_r]), rtol=1e-6)
    io = ours.intensities_km_mol[keep_o][np.argsort(ours.frequencies_cm[keep_o])]
    ir = ref.intensities_km_mol[keep_r][np.argsort(ref.frequencies_cm[keep_r])]
    np.testing.assert_allclose(io, ir, rtol=1e-3, atol=1e-6)


def test_ase_route_equals_ase_own_intensities(triatomic, tmp_path):
    """Our post-processing of ASE's Infrared cache reproduces ASE's own numbers exactly."""
    from ase.vibrations import Infrared

    atoms, calc = triatomic
    a = atoms.copy()
    a.calc = calc
    ours = compute_ir_ase_infrared(a, None, tmp_path / "ir", delta=0.01)
    ref = Infrared(a, delta=0.01, name=str(tmp_path / "ir" / "ir"))
    ref.read()
    order = np.argsort(ours.frequencies_cm)
    np.testing.assert_allclose(ours.intensities_km_mol[order] / KMMOL_PER_D2_A2_AMU,
                               ref.intensities, rtol=1e-8, atol=1e-10)


def test_ase_route_projection_removes_spurious_rotations(triatomic, tmp_path):
    """With a residual gradient (non-stationary geometry) rotations become 'real' modes in the
    unprojected Hessian; projection leaves exactly 3N-6 vibrations."""
    atoms, calc = triatomic
    a = atoms.copy()
    a.positions[1, 1] += 0.05  # stretch one bond -> residual forces
    a.calc = calc
    raw = compute_ir_ase_infrared(a, None, tmp_path / "raw", delta=0.01)
    proj = compute_ir_ase_infrared(a, None, tmp_path / "proj", delta=0.01, project_rigid=True)
    assert len(raw.frequencies_cm) == 9 and len(proj.frequencies_cm) == 3
    assert np.all(proj.frequencies_cm > 100)


def test_projection_removes_six_rigid_modes(triatomic):
    atoms, calc = triatomic
    res = compute_ir(atoms, ASECalculatorForces(calc), None, project_rigid=True)
    assert len(res.frequencies_cm) == 3
    assert np.all(res.frequencies_cm > 100)
    assert res.intensities_km_mol is None and res.route == "frequencies-only"


def test_rigid_basis_rank_linear_vs_nonlinear(diatomic, triatomic):
    lin, _, _ = diatomic
    bent, _ = triatomic
    assert rigid_body_basis(lin.positions, lin.get_masses()).shape[1] == 5
    assert rigid_body_basis(bent.positions, bent.get_masses()).shape[1] == 6


def test_nearly_linear_molecule_keeps_3n_minus_5_modes():
    """A CO2 bent by 0.05 deg (float32 MLIP relaxation noise) is still treated as linear."""
    import numpy as np
    from ase import Atoms

    ang = np.deg2rad(0.05)
    co2 = Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16 * np.cos(ang), 1.16 * np.sin(ang), 0]])
    assert rigid_body_basis(co2.positions, co2.get_masses()).shape[1] == 5
    bent = Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16 * np.cos(0.3), 1.16 * np.sin(0.3), 0]])
    assert rigid_body_basis(bent.positions, bent.get_masses()).shape[1] == 6


def test_hessian_is_symmetric_and_partial(triatomic):
    atoms, calc = triatomic
    idx = [1, 2]
    disps, images = make_displacements(atoms, idx, 0.01, 2)
    assert len(images) == 2 * 3 * len(idx)
    forces = ASECalculatorForces(calc)(images)
    h = assemble_hessian(forces, disps, idx, 0.01, 2)
    assert h.shape == (6, 6)
    np.testing.assert_allclose(h, h.T, atol=1e-10)


def test_surface_selection_rule_keeps_only_normal_component():
    dmu = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    iso = ir_intensities(dmu)
    mssr = ir_intensities(dmu, surface_normal=np.array([0, 0, 1.0]))
    np.testing.assert_allclose(iso, np.array([1, 4, 3]) * KMMOL_PER_D2_A2_AMU)
    np.testing.assert_allclose(mssr, np.array([0, 4, 1]) * KMMOL_PER_D2_A2_AMU)
    np.testing.assert_allclose(ir_intensities(dmu, np.array([0, 0, 1.0]), image_factor=4.0), 4 * mssr)


def test_surface_normal_from_slab_cell():
    from ase.build import fcc111

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    np.testing.assert_allclose(surface_normal_from_cell(slab.cell), [0, 0, 1], atol=1e-12)


def test_unit_constants():
    assert pytest.approx(4.80320, rel=1e-4) == DEBYE_PER_E_ANG
    # wavenumber of a 1 eV/A^2/amu force constant: same conversion as ASE
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    assert eigenvalues_to_cm(np.array([1.0]))[0] == pytest.approx(s / units.invcm)
    assert eigenvalues_to_cm(np.array([-1.0]))[0] < 0  # imaginary -> negative


def test_normal_modes_shape_errors():
    with pytest.raises(ValueError):
        normal_modes(np.eye(6), np.ones(3))
    with pytest.raises(ValueError):
        normal_modes(np.eye(9), np.ones(3), project_rigid=True)
