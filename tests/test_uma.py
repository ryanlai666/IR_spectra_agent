"""UMA path: batching logic and gating with mocks (CPU, no weights); real model under -m gpu."""

from __future__ import annotations

import numpy as np
import pytest

from ir_agent.calculators import uma
from ir_agent.calculators.errors import CalculatorUnavailableError
from ir_agent.config import UMAConfig
from ir_agent.vibrations.evaluators import ASECalculatorForces
from ir_agent.vibrations.finite_diff import make_displacements
from ir_agent.vibrations.ir import compute_ir


class MockPredictor:
    """Stands in for fairchem's MLIPPredictUnit: forces from the toy harmonic calculator."""

    def __init__(self, calc):
        self.calc = calc
        self.batch_sizes = []

    def predict(self, batch):
        self.batch_sizes.append(len(batch))
        forces = []
        for atoms in batch:
            a = atoms.copy()
            a.calc = self.calc
            forces.append(a.get_forces())
        return {"forces": np.concatenate(forces), "energy": np.zeros(len(batch))}


def list_builder(images):
    return list(images)


def test_batched_forces_equal_loop(triatomic):
    atoms, calc = triatomic
    _, images = make_displacements(atoms, range(len(atoms)), 0.01, 2)  # 18 images
    pred = MockPredictor(calc)
    ev = uma.UMABatchForces(pred, "omol", batch_size=8, batch_builder=list_builder)
    f_batched = ev(images)
    f_loop = ASECalculatorForces(calc)(images)
    np.testing.assert_allclose(f_batched, f_loop)
    assert pred.batch_sizes == [8, 8, 2] and ev.n_forward_passes == 3


def test_vram_guard_caps_atoms_per_pass(triatomic):
    atoms, calc = triatomic  # 3 atoms -> 18 displaced images
    _, images = make_displacements(atoms, range(3), 0.01, 2)
    pred = MockPredictor(calc)
    ev = uma.UMABatchForces(pred, "omol", batch_size=24, batch_builder=list_builder, max_atoms_per_batch=15)
    ev(images)
    assert ev.effective_batch_size(3) == 5
    assert pred.batch_sizes == [5, 5, 5, 3]
    assert uma.UMABatchForces(pred, "oc20", batch_size=24, max_atoms_per_batch=180).effective_batch_size(66) == 2


def test_batched_evaluator_sets_omol_charge_spin(triatomic):
    atoms, calc = triatomic
    seen = []

    def builder(images):
        seen.extend(img.info.copy() for img in images)
        return list(images)

    ev = uma.UMABatchForces(MockPredictor(calc), "omol", batch_size=4, charge=0, spin=1, batch_builder=builder)
    ev([atoms])
    assert seen[0] == {"charge": 0, "spin": 1}


def test_uma_hybrid_ir_with_mock_predictor(triatomic):
    """Full hybrid IR with the mocked UMA force evaluator + toy dipoles."""
    from ir_agent.vibrations.evaluators import ASECalculatorDipoles

    atoms, calc = triatomic
    ev = uma.UMABatchForces(MockPredictor(calc), "omol", batch_size=5, batch_builder=list_builder)
    res = compute_ir(atoms, ev, ASECalculatorDipoles(calc), project_rigid=True)
    ref = compute_ir(atoms, ASECalculatorForces(calc), ASECalculatorDipoles(calc), project_rigid=True)
    np.testing.assert_allclose(res.frequencies_cm, ref.frequencies_cm)
    np.testing.assert_allclose(res.intensities_km_mol, ref.intensities_km_mol)
    assert res.force_method.startswith("uma[omol]")


def test_batch_size_mismatch_detected(triatomic):
    atoms, calc = triatomic

    class Bad(MockPredictor):
        def predict(self, batch):
            return {"forces": np.zeros((1, 3))}

    with pytest.raises(RuntimeError, match="Unexpected force array"):
        uma.UMABatchForces(Bad(calc), "omol", batch_builder=list_builder)([atoms, atoms])


def test_task_selection():
    cfg = UMAConfig()
    assert uma.task_for_system("molecule", cfg) == "omol"
    assert uma.task_for_system("surface", cfg) == "oc20"


def test_task_presets_pin_one_task_for_gas_and_surface(cfg):
    """uma-<task> presets use the same task for the gas reference and the adsorbate."""
    from ir_agent.calculators.registry import PRESETS, canonical_preset
    from ir_agent.validation.run import gas_reference_preset, uma_task

    u = UMAConfig()
    for preset, task in uma.UMA_TASK_PRESETS.items():
        assert canonical_preset(preset) == preset
        assert uma.task_for_preset(preset, "molecule", u) == task
        assert uma.task_for_preset(preset, "surface", u) == task
        assert gas_reference_preset(preset, "surface", cfg) == preset
    assert not PRESETS["uma-omol"].periodic  # -> Pt10/Au10 cluster model
    assert PRESETS["uma-oc20"].periodic and PRESETS["uma-omat"].periodic
    # plain 'uma' switches task: its slab shifts must be referenced to a uma-oc20 gas run
    assert uma_task("uma", "molecule", cfg) == "omol"
    assert uma_task("uma", "surface", cfg) == "oc20"
    assert gas_reference_preset("uma", "surface", cfg) == "uma-oc20"
    assert gas_reference_preset("uma", "cluster", cfg) == "uma"
    assert gas_reference_preset("orb-omat", "surface", cfg) == "orb-omat"
    assert uma_task("orb-mol", "molecule", cfg) is None


def test_task_preset_reaches_batched_evaluator(monkeypatch, cfg, triatomic):
    from ir_agent.calculators import registry

    atoms, calc = triatomic
    monkeypatch.setattr(uma, "load_predictor", lambda u: MockPredictor(calc))
    for preset, kind, task in (("uma-omat", "molecule", "omat"), ("uma-oc20", "molecule", "oc20"),
                               ("uma", "surface", "oc20"), ("uma", "molecule", "omol")):
        fe = registry.make_force_evaluator(preset, cfg, atoms=atoms.copy(), system_kind=kind)
        assert fe.task_name == task


def test_gate_message_without_token(monkeypatch, tmp_path):
    monkeypatch.setattr(uma, "fairchem_available", lambda: True)
    monkeypatch.setattr(uma, "hf_token_available", lambda: False)
    # explicit missing path: the real checkpoint may be present in data/checkpoints/
    cfg = UMAConfig(checkpoint_path=str(tmp_path / "missing.pt"))
    with pytest.raises(CalculatorUnavailableError, match="huggingface.co/facebook/UMA"):
        uma.check_uma_ready(cfg)


def test_load_predictor_maps_gated_error(monkeypatch, tmp_path):
    monkeypatch.setattr(uma, "check_uma_ready", lambda *a, **k: None)

    def boom(*a, **k):
        raise OSError("401 Client Error: Cannot access gated repo facebook/UMA")

    monkeypatch.setattr(uma, "_load_predictor_cached", boom)
    with pytest.raises(CalculatorUnavailableError, match="denied"):
        uma.load_predictor(UMAConfig(device="cpu", checkpoint_path=str(tmp_path / "missing.pt")))


def _fake_ckpt(tmp_path, content=b"not-a-real-checkpoint"):
    import hashlib

    f = tmp_path / "uma-s-1p2p1.pt"
    f.write_bytes(content)
    return f, hashlib.md5(content).hexdigest()


def test_local_checkpoint_md5_mismatch_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(uma, "fairchem_available", lambda: True)
    f, _ = _fake_ckpt(tmp_path)
    cfg = UMAConfig(device="cpu", checkpoint_path=str(f), checkpoint_md5="0" * 32)
    with pytest.raises(CalculatorUnavailableError, match="MD5 mismatch"):
        uma.load_predictor(cfg)


def test_local_checkpoint_loads_without_hub(monkeypatch, tmp_path):
    """Local path -> load_predict_unit; get_predict_unit (gated hub) must never be called."""
    monkeypatch.setattr(uma, "fairchem_available", lambda: True)
    monkeypatch.setattr(uma, "hf_token_available", lambda: False)
    f, md5 = _fake_ckpt(tmp_path)
    calls = []
    monkeypatch.setattr(uma, "_load_local_cached", lambda path, device, settings: calls.append(path) or "PU")

    def hub(*a, **k):
        raise AssertionError("hub download attempted")

    monkeypatch.setattr(uma, "_load_predictor_cached", hub)
    cfg = UMAConfig(device="cpu", checkpoint_path=str(f), checkpoint_md5=md5)
    assert uma.load_predictor(cfg) == "PU"
    assert calls == [str(f)]


def test_missing_local_checkpoint_message(monkeypatch, tmp_path):
    monkeypatch.setattr(uma, "fairchem_available", lambda: True)
    cfg = UMAConfig(checkpoint_path=str(tmp_path / "nope.pt"))
    with pytest.raises(CalculatorUnavailableError, match="local UMA checkpoint not found"):
        uma.check_uma_ready(cfg)


@pytest.mark.skipif(not uma.fairchem_available(), reason="fairchem-core not installed")
def test_fairchem_atomicdata_batch_builder_with_mock_predictor(triatomic):
    """Real fairchem AtomicData batching + mocked prediction (no gated weights needed)."""
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)
    build = uma._default_batch_builder("omol", predictor=object())
    batch = build([uma.prepare_atoms_for_task(atoms.copy(), "omol") for _ in range(3)])
    assert int(batch.natoms.sum()) == 9
    assert tuple(batch.pos.shape) == (9, 3)


@pytest.mark.gpu
@pytest.mark.uma
def test_real_uma_water_on_cuda(cfg):
    """Requires CUDA + accepted facebook/UMA access. Records wall time and peak VRAM."""
    import time

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import os

    from ir_agent.gpu import apply_gpu_guard_cap, guard_reservation_gb, guard_script

    if (guard_script() and not guard_reservation_gb()
            and os.environ.get("IR_AGENT_DEDICATED_GPU") != "1"):
        pytest.skip("GPU_GUARD is set: launch this test through the guard (it sets GPU_GUARD_GB)")
    apply_gpu_guard_cap()
    try:
        uma.check_uma_ready(cfg.calculators.uma)
    except CalculatorUnavailableError as exc:
        pytest.skip(str(exc).splitlines()[0])
    from ase.build import molecule

    from ir_agent.calculators.registry import make_dipole_evaluator, preset_status

    u = cfg.calculators.uma.model_copy(update={"device": "cuda"})
    pred = uma.load_predictor(u)
    atoms = molecule("H2O")
    atoms.center(vacuum=6.0)
    atoms.info.update(charge=0, spin=1)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    de = make_dipole_evaluator("xtb", cfg) if preset_status("xtb", cfg)[0] else None
    res = compute_ir(atoms, uma.UMABatchForces(pred, "omol", u.batch_size), de, project_rigid=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"UMA H2O hybrid IR: {wall:.2f} s, peak VRAM {peak:.2f} GiB, freqs {res.frequencies_cm}")
    assert peak < 8.0
    assert len(res.frequencies_cm) == 3
