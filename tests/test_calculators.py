from __future__ import annotations

import pytest
from ase.build import molecule

from ir_agent.calculators.errors import CalculatorUnavailableError, UnsupportedSystemError
from ir_agent.calculators.registry import (
    PRESETS,
    canonical_preset,
    list_presets,
    make_calculator,
    preset_status,
    resolve_dipole_method,
)
from ir_agent.structures import build_slab


def test_every_preset_reports_status(cfg):
    rows = list_presets(cfg)
    assert {r["name"] for r in rows} == set(PRESETS)
    assert next(r for r in rows if r["name"] == "emt")["available"]


def test_unknown_preset():
    with pytest.raises(ValueError, match="Unknown calculator"):
        canonical_preset("magic")


def test_missing_calculator_raises_clear_error(cfg, monkeypatch):
    monkeypatch.delenv("VASP_COMMAND", raising=False)
    monkeypatch.delenv("ASE_VASP_COMMAND", raising=False)
    with pytest.raises(CalculatorUnavailableError, match="VASP"):
        make_calculator("vasp", cfg, atoms=molecule("H2O"))


def test_emt_rejects_unsupported_elements(cfg):
    with pytest.raises(UnsupportedSystemError, match="EMT"):
        make_calculator("emt", cfg, atoms=molecule("CH3Cl"))


def test_xtb_rejects_periodic_slab(cfg):
    if not preset_status("xtb", cfg)[0]:
        pytest.skip("xtb not installed")
    with pytest.raises(UnsupportedSystemError, match="periodic"):
        make_calculator("xtb", cfg, atoms=build_slab("Pt", "fcc", "111", (2, 2, 2)))


def test_dipole_resolution(cfg, monkeypatch):
    assert resolve_dipole_method("none", "emt", cfg) is None
    with pytest.raises(UnsupportedSystemError):
        resolve_dipole_method("native", "emt", cfg)
    # with no dipole backend installed, auto -> None (frequencies only)
    import ir_agent.calculators.registry as reg

    monkeypatch.setattr(reg, "preset_status", lambda name, c: (name == "emt", "patched"))
    assert resolve_dipole_method("auto", "emt", cfg) is None
    monkeypatch.setattr(reg, "preset_status", lambda name, c: (name in ("emt", "dxtb"), "patched"))
    assert resolve_dipole_method("auto", "emt", cfg) == "dxtb"
