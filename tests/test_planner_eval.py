"""Planner-accuracy harness: labels are valid plans, and scoring works with a fake LLM."""

from __future__ import annotations

from pathlib import Path

from ir_agent.agent.plan import PlannerOutput, SimulationPlan, SurfaceSpec
from ir_agent.agent.planner_eval import (
    evaluate,
    load_cases,
    plan_fields,
    render_markdown,
    score_case,
)
from ir_agent.calculators.registry import canonical_preset
from ir_agent.config import LLMConfig

CASES = Path(__file__).resolve().parents[1] / "examples" / "planner_eval" / "prompts.jsonl"


def _oracle(expected: dict) -> dict:
    """PlannerOutput dict that satisfies ``expected`` (first acceptable calculator)."""
    calc = expected.get("calculator", "emt")
    out = {"adsorbate": expected["adsorbate"], "calculator": calc[0] if isinstance(calc, list) else calc,
           "surface_metal": expected.get("metal") or "None"}
    for f, k in (("structure", "surface_structure"), ("facet", "surface_facet"), ("site", "surface_site"),
                 ("model", "surface_model"), ("dipole_method", "dipole_method")):
        if f in expected:
            out[k] = expected[f]
    if expected.get("structure") == "hcp" and "facet" not in expected:
        out["surface_facet"] = "0001"
    return out


def test_cases_are_valid_labels():
    cases = load_cases(CASES)
    assert 10 <= len(cases) <= 20
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        e = c["expected"]
        for calc in e.get("calculator", []) if isinstance(e.get("calculator"), list) else [e.get("calculator")]:
            if calc is not None:
                assert canonical_preset(calc) == calc, c["id"]
        plan = PlannerOutput(**_oracle(e)).to_plan(c["prompt"])  # raises if the label is impossible
        assert all(score_case(e, plan).values()), c["id"]


def test_evaluate_with_fake_llm_scores_misses():
    cases = load_cases(CASES)[:4]
    wrong_id = cases[1]["id"]

    def fake_invoke(model, messages):
        prompt = messages[1][1].removeprefix("User task: ")
        case = next(c for c in cases if c["prompt"] == prompt)
        out = _oracle(case["expected"])
        if case["id"] == wrong_id:
            out["calculator"] = "emt"  # a planner mistake
        return out

    res = evaluate(cases, LLMConfig(max_retries=1, fallback_models=[]), fake_invoke)
    m = res["metrics"]
    assert res["n"] == 4 and m["valid_plan_rate"] == 1.0
    assert m["exact_match"] == 0.75
    assert m["field_accuracy"]["calculator"] == 0.75 and m["field_accuracy"]["adsorbate"] == 1.0
    assert m["first_try_rate"] == 1.0
    md = render_markdown(res)
    assert wrong_id in md and "calculator: expected" in md


def test_invalid_plan_counts_as_miss():
    def broken(model, messages):
        return {"adsorbate": "Unobtainium", "calculator": "emt", "surface_metal": "None"}

    case = {"id": "x", "prompt": "nonsense", "expected": {"adsorbate": "H2O", "metal": None}}
    res = evaluate([case], LLMConfig(max_retries=2, fallback_models=[]), broken)
    row = res["rows"][0]
    assert not row["valid"] and not row["exact"] and row["error"]
    assert res["metrics"]["valid_plan_rate"] == 0.0
    assert res["metrics"]["validator_rejections"] == 2 and res["metrics"]["first_try_rate"] is None


def test_api_errors_are_not_validator_rejections():
    from ir_agent.agent.planner_eval import metrics

    row = {"scores": {"adsorbate": True}, "exact": True, "valid": True, "attempts": 2, "latency_s": 1.0,
           "model": "m2", "api_errors": ["m1#1: 503 UNAVAILABLE", "m2#1: 429 RESOURCE_EXHAUSTED"]}
    m = metrics([row, {**row, "api_errors": ["m2#1: invalid plan: surface_metal: bad"]}])
    assert m["first_try_rate"] == 0.5 and m["validator_rejections"] == 1
    assert m["answered_by"] == {"m2": 2} and m["rows_with_api_errors"] == 2


def test_plan_fields_and_site_normalization():
    plan = PlannerOutput(adsorbate="CO", calculator="uma-oc20", surface_metal="Pt",
                         surface_site="hollow", surface_model="slab").to_plan()
    assert plan_fields(plan)["site"] == "fcc" and plan.surface.model == "slab"
    gas = SimulationPlan(adsorbate="H2O", calculator="xtb")
    assert plan_fields(gas)["metal"] is None
    assert score_case({"metal": None, "calculator": ["xtb", "tblite"]}, gas) == {"metal": True, "calculator": True}
    assert SurfaceSpec(metal="pt").metal == "Pt"
