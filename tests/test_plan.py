from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ir_agent.agent.llm import PlanningError, plan_with_llm
from ir_agent.agent.plan import PlannerOutput, SimulationPlan, load_plan, plan_from_dict
from ir_agent.config import LLMConfig


def test_nested_plan_roundtrip(tmp_path):
    plan = SimulationPlan(adsorbate="CO", calculator="uma",
                          surface={"metal": "pt", "structure": "fcc", "facet": "(111)"})
    assert plan.surface.metal == "Pt" and plan.surface.facet == "111"
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan.model_dump()))
    assert load_plan(p) == plan
    assert plan.label == "CO_on_Pt111_fcc_uma"


def test_aliases_are_normalized():
    assert SimulationPlan(adsorbate="H2O", calculator="fast").calculator == "emt"
    assert SimulationPlan(adsorbate="H2O", calculator="GFN2").calculator == "xtb"
    assert SimulationPlan(adsorbate="H2O", calculator="accurate").calculator == "vasp-accurate"


@pytest.mark.parametrize(
    "bad",
    [
        {"adsorbate": "Unobtainium", "calculator": "emt"},
        {"adsorbate": "CO", "calculator": "quantum-magic"},
        {"adsorbate": "CO", "surface": {"metal": "Pt", "structure": "fcc", "facet": "0001"}},
        {"adsorbate": "CO", "surface": {"metal": "Qq"}},
        {"adsorbate": "CO", "spin_multiplicity": 0},
    ],
)
def test_invalid_plans_rejected(bad):
    with pytest.raises((ValidationError, ValueError)):
        SimulationPlan.model_validate(bad)


def test_legacy_v01_plan_keys():
    plan = plan_from_dict({
        "calculator_level": "fast", "surface_metal": "Au", "adsorbate_formula": "H2O",
        "surface_metal_structure_type": "fcc", "surface_metal_facet": "100",
    })
    assert plan.calculator == "emt" and plan.surface.metal == "Au" and plan.surface.facet == "100"
    gas = plan_from_dict({"calculator_level": "xtb", "surface_metal": "None", "adsorbate_formula": "H2O"})
    assert gas.surface is None


def test_planner_output_to_plan_gas_phase():
    out = PlannerOutput(calculator="xtb", adsorbate="NH3", surface_metal="none")
    assert out.to_plan("x").surface is None


def test_llm_retry_feeds_validation_error_back():
    calls = []

    def fake(model, messages):
        calls.append((model, list(messages)))
        if len(calls) == 1:  # invalid: unknown facet for fcc
            return {"calculator": "emt", "adsorbate": "CO", "surface_metal": "Pt",
                    "surface_structure": "fcc", "surface_facet": "0001"}
        return {"calculator": "emt", "adsorbate": "CO", "surface_metal": "Pt",
                "surface_structure": "fcc", "surface_facet": "111"}

    plan, meta = plan_with_llm("CO on Pt", LLMConfig(model="m1", fallback_models=[]), fake)
    assert plan.surface.facet == "111"
    assert meta["attempts"] == 2
    feedback = calls[1][1][-1][1]
    assert "rejected by the validator" in feedback and "facet" in feedback


def test_llm_falls_back_when_model_retired():
    def fake(model, messages):
        if model == "gemini-old":
            raise RuntimeError("404 NOT_FOUND: models/gemini-old is not found")
        return PlannerOutput(calculator="xtb", adsorbate="H2O", surface_metal="None")

    plan, meta = plan_with_llm("water", LLMConfig(model="gemini-old", fallback_models=["gemini-new"]), fake)
    assert meta["model"] == "gemini-new" and plan.calculator == "xtb"


def test_llm_gives_up_after_retries():
    def fake(model, messages):
        return {"calculator": "emt", "adsorbate": "NotAMolecule", "surface_metal": "None"}

    with pytest.raises(PlanningError, match="invalid plan"):
        plan_with_llm("x", LLMConfig(model="m", fallback_models=[], max_retries=2), fake)
