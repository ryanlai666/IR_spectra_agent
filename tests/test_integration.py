"""End-to-end CPU runs (< 2 min total): EMT always; xTB when the binary is installed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ir_agent.agent.graph import build_graph
from ir_agent.agent.plan import SimulationPlan, SurfaceSpec
from ir_agent.calculators.registry import preset_status
from ir_agent.pipeline import run_pipeline
from ir_agent.runtime import RunDir

pytestmark = pytest.mark.integration


def _fast(cfg):
    return cfg.model_copy(update={
        "geometry": cfg.geometry.model_copy(update={"slab_size": (2, 2, 3)}),
        "optimization": cfg.optimization.model_copy(update={"max_steps": 150}),
    })


def test_graph_no_llm_emt_gas_phase(cfg, tmp_path):
    """Deterministic --no-llm graph run writes plan, report, plots and JSON."""
    app = build_graph()
    plan = SimulationPlan(adsorbate="CO", calculator="emt", dipole_method="none")
    final = app.invoke({"config": _fast(cfg), "plan": plan, "use_llm": False, "task": "CO gas EMT"})
    assert not final.get("error"), final.get("error")
    arts = final["artifacts"]
    for key in ("report_md", "spectrum_png", "structures_png"):
        assert arts[key].endswith((".md", ".png"))
    run_dir = RunDir(__import__("pathlib").Path(arts["run_dir"]))
    plan_back = json.loads((run_dir.root / "plan.json").read_text())
    assert plan_back["adsorbate"] == "CO"
    res = final["ir_result"]
    assert res.route == "frequencies-only" and len(res.frequencies_cm) == 1


def test_graph_failure_route(cfg):
    """Unavailable calculator -> failure node, error recorded, no exception."""
    app = build_graph()
    plan = SimulationPlan(adsorbate="H2O", calculator="vasp")
    final = app.invoke({"config": cfg, "plan": plan, "use_llm": False})
    assert final["failed_node"] == "geometry"
    assert "unavailable" in final["error"]
    assert "Run failed" in final["report_markdown"]


def test_graph_llm_path_with_fake_invokers(cfg):
    app = build_graph()

    def structured(model, messages):
        return {"calculator": "fast", "adsorbate": "CO", "surface_metal": "None"}

    def text(model, prompt):
        assert "IR report" in prompt
        return "Mock narrative."

    final = app.invoke({"config": _fast(cfg), "task": "CO please", "use_llm": True,
                        "llm_invokers": (structured, text)})
    assert not final.get("error"), final.get("error")
    assert final["plan"].calculator == "emt"
    assert "Mock narrative." in final["report_markdown"]


def test_emt_slab_hybrid_with_mssr(cfg):
    """CO on Pt(111) slab with EMT forces; dipoles from xTB fragment if available."""
    plan = SimulationPlan(adsorbate="CO", calculator="emt",
                          surface=SurfaceSpec(metal="Pt", structure="fcc", facet="111"))
    out = run_pipeline(plan, _fast(cfg))
    res = out["result"]
    assert out["prepared"].kind == "surface"
    assert len(res.frequencies_cm) == 6  # partial Hessian of the 2 adsorbate atoms
    if res.has_intensities:
        assert res.surface_selection_rule
        assert np.all(res.intensities_km_mol <= res.intensities_isotropic_km_mol + 1e-9)


@pytest.mark.skipif("not __import__('ir_agent.calculators.xtb_binary', fromlist=['x']).find_xtb_binary()")
def test_xtb_water_matches_nist_and_native_hessian(cfg):
    """GFN2-xTB H2O: xtb --hess route (+ FD native dipoles) vs xtb's own IR and NIST."""
    from ir_agent.analysis import match_gas_phase
    from ir_agent.calculators.xtb_binary import XTBBinary

    assert preset_status("xtb", cfg)[0]
    plan = SimulationPlan(adsorbate="H2O", calculator="xtb")
    out = run_pipeline(plan, cfg)
    res = out["result"]
    assert res.route == "xtb-hessian"
    matches = match_gas_phase("H2O", res)
    errs = {m.mode: m.error_cm for m in matches}
    assert abs(errs["bend"]) < 100  # GFN2 bend is close to experiment
    native_f, native_i = XTBBinary().native_ir(out["prepared"].stages["final"])
    keep = res.frequencies_cm > 100
    order = np.argsort(res.frequencies_cm[keep])
    f, inten = res.frequencies_cm[keep][order], res.intensities_km_mol[keep][order]
    nkeep = native_f > 100
    norder = np.argsort(native_f[nkeep])
    np.testing.assert_allclose(f, native_f[nkeep][norder], atol=1.0)
    # same Hessian, dipole derivatives by finite differences vs xtb's analytic ones
    np.testing.assert_allclose(inten, native_i[nkeep][norder], rtol=0.05, atol=1.0)
