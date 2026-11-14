"""Regression tests for the xtb-executable Hessian route.

Finite differences of xtb 6.7.1 ``--grad`` gradients split the degenerate modes of NH3 (GFN1
deg. deformation 1235/1610 cm^-1) and CO2 (bend 570/628), because that single-point gradient is
not rotationally invariant for slightly distorted near-symmetric geometries. The pipeline now
uses xtb's own ``--hess`` Hessian; these tests pin degeneracy and agreement with xtb's IR.
"""

from __future__ import annotations

import numpy as np
import pytest

from ir_agent.agent.plan import SimulationPlan
from ir_agent.calculators.xtb_binary import _read_hessian, find_xtb_binary
from ir_agent.pipeline import run_pipeline

needs_xtb = pytest.mark.skipif(find_xtb_binary() is None, reason="xtb executable not installed")


def test_read_hessian_parses_turbomole_block(tmp_path):
    h = np.arange(36, dtype=float).reshape(6, 6)
    h = h + h.T
    lines = ["$hessian"]
    flat = h.ravel()
    for k in range(0, flat.size, 5):  # xtb writes 5 values per line
        lines.append("  ".join(f"{x:.10f}" for x in flat[k:k + 5]))
    lines.append("$end")
    p = tmp_path / "hessian"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.testing.assert_allclose(_read_hessian(p, 2), h)


def _modes(cfg, mol, calc):
    out = run_pipeline(SimulationPlan(adsorbate=mol, calculator=calc), cfg)
    res = out["result"]
    assert res.route == "xtb-hessian"
    f = np.sort(res.frequencies_cm[res.frequencies_cm > 100])
    return f, out


@needs_xtb
@pytest.mark.integration
@pytest.mark.parametrize("calc", ["xtb-gfn1", "xtb"])
def test_nh3_degenerate_pairs_stay_degenerate(cfg, calc):
    from ir_agent.calculators.xtb_binary import XTBBinary

    f, out = _modes(cfg, "NH3", calc)
    assert len(f) == 6  # umbrella, deg. deformation x2, sym. stretch, deg. stretch x2
    assert abs(f[1] - f[2]) < 1.0, f  # deg. deformation (E)
    assert abs(f[4] - f[5]) < 1.0, f  # deg. NH stretch (E)
    native, _ = XTBBinary(method="gfn1" if calc == "xtb-gfn1" else "gfn2").native_ir(
        out["prepared"].stages["final"])
    np.testing.assert_allclose(f, np.sort(native[native > 100]), atol=1.0)


@needs_xtb
@pytest.mark.integration
def test_co2_bend_degenerate_and_matches_native(cfg):
    from ir_agent.calculators.xtb_binary import XTBBinary

    f, out = _modes(cfg, "CO2", "xtb-gfn1")
    assert len(f) == 4  # bend x2, sym. stretch, asym. stretch (3N-5)
    assert abs(f[0] - f[1]) < 1.0, f
    native, _ = XTBBinary(method="gfn1").native_ir(out["prepared"].stages["final"])
    np.testing.assert_allclose(f, np.sort(native[native > 100]), atol=1.0)
