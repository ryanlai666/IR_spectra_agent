"""Orb (orb-models) and PET (upet) presets: mocked unit tests + one smoke test per family.

The unit tests need neither package. The smoke tests run only where the package and the local
checkpoint exist (.venv-orb / .venv-pet) and are skipped elsewhere (main venv, CI).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from ir_agent.calculators import orb, pet
from ir_agent.calculators.errors import CalculatorUnavailableError
from ir_agent.calculators.registry import PRESETS, canonical_preset, make_calculator, preset_status
from ir_agent.config import OrbConfig, PETConfig
from ir_agent.vibrations.evaluators import ASECalculatorForces
from ir_agent.vibrations.finite_diff import make_displacements


def test_presets_and_aliases():
    assert canonical_preset("orb") == "orb-mol" and canonical_preset("orbmol-v2") == "orb-mol"
    assert canonical_preset("pet") == "pet-omol" and canonical_preset("pet-mad-s") == "pet-mad"
    # molecular models -> cluster model on surfaces; materials models -> periodic slab
    assert not PRESETS["orb-mol"].periodic and PRESETS["orb-omat"].periodic
    assert not PRESETS["pet-omol"].periodic and PRESETS["pet-mad"].periodic
    assert all(not PRESETS[k].native_dipole for k in ("orb-mol", "orb-omat", "pet-omol", "pet-mad"))


def test_unavailable_gives_install_hint(cfg, monkeypatch):
    monkeypatch.setattr(orb, "orb_available", lambda: False)
    monkeypatch.setattr(pet, "pet_available", lambda: False)
    assert preset_status("orb-mol", cfg)[0] is False
    assert preset_status("pet-omol", cfg)[0] is False
    with pytest.raises(CalculatorUnavailableError, match=r"\.venv-orb"):
        make_calculator("orb-mol", cfg, atoms=molecule("H2O"))
    with pytest.raises(CalculatorUnavailableError, match=r"\.venv-pet"):
        make_calculator("pet-omol", cfg, atoms=molecule("H2O"))


def test_pet_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(pet, "pet_available", lambda: True)
    ok, reason = pet.status("pet-omol", PETConfig(models_dir=str(tmp_path)))
    assert not ok and "checkpoint missing" in reason


def test_charge_spin_handling():
    a = molecule("H2O")
    orb.prepare_atoms(a, "orb-mol", charge=0, spin=1)
    assert a.info["charge"] == 0 and a.info["spin"] == 1
    orb.prepare_atoms(a, "orb-omat")  # bulk model must not get OMol conditioning inputs
    assert "charge" not in a.info and "spin" not in a.info
    b = molecule("NO")
    pet.prepare_atoms(b, "pet-omol", charge=0, spin=2)
    assert b.info == {"charge": 0, "spin": 2}


def test_orb_rejects_non_conservative(monkeypatch):
    class DirectForcefieldRegressor:  # mimics orb's direct (non-conservative) model class
        pass

    monkeypatch.setattr(orb, "orb_available", lambda: True)
    monkeypatch.setattr(orb, "_load_cached", lambda *a: (DirectForcefieldRegressor(), object()))
    with pytest.raises(CalculatorUnavailableError, match="not a conservative model"):
        orb.load_model("orb-mol", OrbConfig(weights_dir=None))


def test_orb_batched_forces_with_mock(triatomic):
    """Batching, charge/spin injection and the VRAM cap, with a mocked Orb predict."""
    atoms, calc = triatomic
    seen_info, batches = [], []

    def fake_predict(images):
        batches.append(len(images))
        seen_info.extend(dict(img.info) for img in images)
        out = []
        for img in images:
            a = img.copy()
            a.calc = calc
            out.append(a.get_forces())
        return np.concatenate(out)

    cfg = OrbConfig(batch_size=16, max_atoms_per_batch=21)  # 21 // 3 atoms -> 7 per pass
    ev = orb.OrbBatchForces("orb-mol", None, None, cfg, charge=0, spin=1, predict=fake_predict)
    _, images = make_displacements(atoms, range(3), 0.01, 2)  # 18 images
    np.testing.assert_allclose(ev(images), ASECalculatorForces(calc)(images))
    assert batches == [7, 7, 4]
    assert all(i.get("charge") == 0 and i.get("spin") == 1 for i in seen_info)
    assert ev.name.startswith("orb-mol[orbmol-v2]")


# ------------------------------------------------------------------------------- smoke tests


def _orb_ready() -> bool:
    return orb.orb_available() and orb._weights_path("orbmol-v2", OrbConfig()) is not None


def _pet_ready() -> bool:
    cfg = PETConfig(molecule_checkpoint="pet-omol-s-v1.0.0.ckpt")
    return pet.status("pet-omol", cfg)[0]


@pytest.mark.integration
@pytest.mark.skipif(not _orb_ready(), reason="orb-models / orbmol-v2 weights not available (use .venv-orb)")
def test_smoke_orb_mol_water_ir(cfg):
    from ir_agent.agent.plan import SimulationPlan
    from ir_agent.analysis import match_gas_phase
    from ir_agent.pipeline import run_pipeline

    out = run_pipeline(SimulationPlan(adsorbate="H2O", calculator="orb-mol"), cfg)
    res = out["result"]
    assert res.route == "hybrid" and res.force_method.startswith("orb-mol")
    m = {x.mode: x for x in match_gas_phase("H2O", res)}
    assert 1500 < m["bend"].calc_cm < 1750 and 3600 < m["OH asym. stretch"].calc_cm < 4100
    if res.has_intensities:
        assert m["bend"].calc_intensity > m["OH sym. stretch"].calc_intensity


@pytest.mark.integration
@pytest.mark.skipif(not _pet_ready(), reason="upet / pet-omol-s checkpoint not available (use .venv-pet)")
def test_smoke_pet_omol_water_ir(cfg):
    from ir_agent.agent.plan import SimulationPlan
    from ir_agent.analysis import match_gas_phase
    from ir_agent.pipeline import run_pipeline

    small = cfg.model_copy(update={"calculators": cfg.calculators.model_copy(update={
        "pet": cfg.calculators.pet.model_copy(update={"molecule_checkpoint": "pet-omol-s-v1.0.0.ckpt"})})})
    out = run_pipeline(SimulationPlan(adsorbate="H2O", calculator="pet-omol"), small)
    m = {x.mode: x for x in match_gas_phase("H2O", out["result"])}
    assert 1500 < m["bend"].calc_cm < 1750 and 3600 < m["OH asym. stretch"].calc_cm < 4100
