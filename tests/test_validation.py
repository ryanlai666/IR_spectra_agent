from __future__ import annotations

import numpy as np
import pytest

from ir_agent.analysis import error_stats, match_gas_phase, optimal_scale_factor
from ir_agent.validation import run as vrun
from ir_agent.validation.reference import ADSORBATE_REFERENCES, GAS_PHASE, expanded_fundamentals
from ir_agent.vibrations.ir import IRResult


def _fake_result(freqs, ints=None):
    f = np.asarray(freqs, dtype=float)
    return IRResult(frequencies_cm=f, intensities_km_mol=None if ints is None else np.asarray(ints, float))


def test_reference_tables_consistent():
    # 3N-6 (3N-5 linear) fundamentals incl. degeneracy
    assert len(expanded_fundamentals("H2O")) == 3
    assert len(expanded_fundamentals("NH3")) == 6
    assert len(expanded_fundamentals("CO2")) == 4
    assert len(expanded_fundamentals("CO")) == 1
    assert set(GAS_PHASE) == {"H2O", "CO", "NH3", "CO2"}
    assert ADSORBATE_REFERENCES[("CO", "Pt")].adsorbed_cm < ADSORBATE_REFERENCES[("CO", "Pt")].gas_cm


def test_match_gas_phase_degenerate_averaging():
    # NH3: computed 6 modes (two near-degenerate pairs) + one spurious low mode
    res = _fake_result([20, 1000, 1600, 1610, 3400, 3500, 3510], [0, 100, 10, 12, 1, 2, 3])
    m = {x.mode: x for x in match_gas_phase("NH3", res)}
    assert m["deg. deformation"].calc_cm == pytest.approx(1605)
    assert m["deg. deformation"].calc_intensity == pytest.approx(22)
    assert m["NH deg. stretch"].calc_cm == pytest.approx(3505)
    assert m["sym. deformation (umbrella)"].error_cm == pytest.approx(50)


def test_match_returns_none_when_too_few_modes():
    assert match_gas_phase("H2O", _fake_result([1500, 3600])) is None


def test_scale_factor_and_stats():
    res = _fake_result([1595 / 0.95, 3657 / 0.95, 3756 / 0.95])
    ms = match_gas_phase("H2O", res)
    assert optimal_scale_factor(ms) == pytest.approx(0.95)
    assert error_stats(ms, 0.95)["mae_cm"] == pytest.approx(0.0, abs=1e-9)


def test_validation_resume_skips_finished_runs(cfg, monkeypatch):
    cfg = cfg.model_copy(update={"ir": cfg.ir.model_copy(update={"dipole_method": "none"})})
    first = vrun.run_validation(cfg, ["emt"], ["CO"], [], resume_dir=None)
    assert first["gas"][0]["molecule"] == "CO" and "error" not in first["gas"][0]

    def boom(*a, **k):
        raise AssertionError("should have been resumed, not recomputed")

    monkeypatch.setattr(vrun, "run_pipeline", boom)
    second = vrun.run_validation(cfg, ["emt"], ["CO"], [], resume_dir=first["dir"])
    assert second["gas"][0]["matches"] == first["gas"][0]["matches"]
    assert "| CO | C-O stretch |" in second["markdown"]
