"""Planner accuracy: run the LLM planner on labelled prompts and score the plans field by field.

Cases live in ``examples/planner_eval/prompts.jsonl``, one JSON object per line:

    {"id": "co-pt-uma", "prompt": "CO on Pt(111) with UMA",
     "expected": {"adsorbate": "CO", "calculator": "uma", "metal": "Pt", "facet": "111"}}

Only the fields present in ``expected`` are scored. ``calculator`` may be a list of acceptable
presets; ``metal: null`` means gas phase. Scored fields: adsorbate, calculator, metal, structure,
facet, site, model, dipole_method. ``scripts/eval_planner.py`` runs it against Gemini.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ir_agent.agent.plan import SimulationPlan
from ir_agent.config import LLMConfig

FIELDS = ("adsorbate", "calculator", "metal", "structure", "facet", "site", "model", "dipole_method")


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.lstrip().startswith("#"):
            cases.append(json.loads(ln))
    return cases


def plan_fields(plan: SimulationPlan) -> dict[str, Any]:
    s = plan.surface
    return {
        "adsorbate": plan.adsorbate,
        "calculator": plan.calculator,
        "metal": s.metal if s else None,
        "structure": s.structure if s else None,
        "facet": s.facet if s else None,
        "site": s.site if s else None,
        "model": s.model if s else None,
        "dipole_method": plan.dipole_method,
    }


def score_case(expected: dict[str, Any], plan: SimulationPlan | None) -> dict[str, bool]:
    """Field -> correct? for every field named in ``expected`` (all False if there is no plan)."""
    got = plan_fields(plan) if plan is not None else {}
    out: dict[str, bool] = {}
    for f, want in expected.items():
        if f not in FIELDS:
            raise KeyError(f"unknown expected field '{f}'")
        have = got.get(f) if plan is not None else object()
        ok_values = want if isinstance(want, list) else [want]
        out[f] = have in ok_values
    return out


def evaluate(
    cases: list[dict[str, Any]],
    cfg: LLMConfig,
    invoke: Callable[[str, list[tuple[str, str]]], Any],
    planner: Callable[..., tuple[SimulationPlan, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run the planner on every case (with the production retry/validation loop)."""
    if planner is None:
        from ir_agent.agent.llm import plan_with_llm as planner
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        plan, meta, err = None, {}, None
        try:
            plan, meta = planner(c["prompt"], cfg, invoke)
        except Exception as exc:  # PlanningError and API failures count as an invalid plan
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            # PlanningError lists one "model#attempt: reason" line per failed attempt
            meta = {"errors": [ln.strip() for ln in str(exc).splitlines()[1:] if ln.strip()]}
        scores = score_case(c["expected"], plan)
        rows.append({
            "id": c["id"], "prompt": c["prompt"], "expected": c["expected"],
            "got": plan_fields(plan) if plan else None, "scores": scores,
            "exact": bool(scores) and all(scores.values()), "valid": plan is not None,
            "attempts": meta.get("attempts"), "model": meta.get("model"), "error": err,
            "api_errors": [e[:200] for e in meta.get("errors", [])],
            "latency_s": time.perf_counter() - t0,
        })
    return {"model": cfg.model, "fallbacks": list(cfg.fallback_models), "n": len(rows),
            "rows": rows, "metrics": metrics(rows)}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_field: dict[str, list[bool]] = {}
    for r in rows:
        for f, ok in r["scores"].items():
            per_field.setdefault(f, []).append(ok)
    n = len(rows) or 1
    # validator rejections only; API errors (503/429) before an answer are not planner mistakes
    rejected = [sum("invalid plan" in e for e in r.get("api_errors", [])) for r in rows]
    valid = [k for k, r in enumerate(rows) if r["valid"]]
    return {
        "exact_match": sum(r["exact"] for r in rows) / n,
        "valid_plan_rate": sum(r["valid"] for r in rows) / n,
        "field_accuracy": {f: sum(v) / len(v) for f, v in per_field.items()},
        "field_counts": {f: len(v) for f, v in per_field.items()},
        "first_try_rate": (sum(rejected[k] == 0 for k in valid) / len(valid)) if valid else None,
        "validator_rejections": sum(rejected),
        "mean_latency_s": sum(r["latency_s"] for r in rows) / n,
        "answered_by": dict(Counter(r["model"] for r in rows if r.get("model"))),
        "rows_with_api_errors": sum(bool(r.get("api_errors")) for r in rows),
    }


def render_markdown(result: dict[str, Any]) -> str:
    m = result["metrics"]
    first = "-" if m["first_try_rate"] is None else f"{m['first_try_rate']:.0%}"
    answered = ", ".join(f"{k} {v}" for k, v in sorted(m.get("answered_by", {}).items())) or "-"
    fallbacks = ", ".join(result.get("fallbacks", [])) or "none"
    lines = [f"# Planner accuracy ({result['model']}, {result['n']} prompts)", "",
             f"- Configured model: {result['model']} (fallbacks: {fallbacks})",
             f"- Answered by: {answered}"
             + (f"; {m['rows_with_api_errors']} prompts saw API errors before an answer"
                if m.get("rows_with_api_errors") else ""),
             f"- Exact plan match: {m['exact_match']:.0%}",
             f"- Valid plan after the validator/retry loop: {m['valid_plan_rate']:.0%}",
             f"- Valid without a validator rejection: {first}"
             f" ({m.get('validator_rejections', 0)} rejections in total)",
             f"- Mean latency: {m['mean_latency_s']:.1f} s", "",
             "| field | accuracy | n |", "|---|---:|---:|"]
    for f in FIELDS:
        if f in m["field_accuracy"]:
            lines.append(f"| {f} | {m['field_accuracy'][f]:.0%} | {m['field_counts'][f]} |")
    lines += ["", "## Misses", ""]
    misses = [r for r in result["rows"] if not r["exact"]]
    if not misses:
        lines.append("None.")
    for r in misses:
        wrong = [f for f, ok in r["scores"].items() if not ok]
        got = r["got"] or {}
        detail = ", ".join(f"{f}: expected {r['expected'][f]!r}, got {got.get(f)!r}" for f in wrong)
        lines.append(f"- `{r['id']}` \"{r['prompt']}\": {detail}" + (f" ({r['error']})" if r["error"] else ""))
    return "\n".join(lines) + "\n"
