"""LangGraph state and node functions.

PLANNER -> GEOMETRY -> CALCULATION -> CRITIC -> PLOT_GEOMETRY -> SPECTRA -> ANALYSIS, where any
node that raises routes to FAILURE (writes an error report) instead of crashing the graph. CRITIC
loops back to CALCULATION (at most ``critic.max_retries`` times) for non-convergence or imaginary
modes and flags site changes / desorption in the report.
"""

from __future__ import annotations

import functools
import traceback
from collections.abc import Callable
from typing import Any, TypedDict

from ir_agent.agent.plan import SimulationPlan, plan_from_dict
from ir_agent.config import AppConfig
from ir_agent.runtime import RunDir, get_logger, setup_logging

log = get_logger("agent")


class SimulationState(TypedDict, total=False):
    task: str
    config: AppConfig
    use_llm: bool
    llm_invokers: Any  # (structured_invoke, text_invoke) - injectable for tests
    plan: SimulationPlan
    plan_meta: dict[str, Any]
    run_dir: RunDir
    prepared: Any  # ir_agent.pipeline.PreparedSystem
    ir_result: Any  # ir_agent.vibrations.ir.IRResult
    critic: dict[str, Any]  # latest ir_agent.critic.Critique.as_dict()
    critic_history: list[dict[str, Any]]
    critic_attempts: int
    retry: bool
    spectrum: Any  # ir_agent.spectrum.Spectrum
    artifacts: dict[str, str]
    report_markdown: str
    summary: dict[str, Any]
    narrative: str
    error: str
    failed_node: str


def _guard(name: str) -> Callable:
    """Catch exceptions in a node and record them in the state for the FAILURE route."""

    def deco(fn: Callable[[SimulationState], dict[str, Any]]):
        @functools.wraps(fn)
        def wrapper(state: SimulationState) -> dict[str, Any]:
            log.info("--- node: %s ---", name)
            try:
                return fn(state)
            except Exception as exc:
                log.error("Node '%s' failed: %s", name, exc)
                log.debug("%s", traceback.format_exc())
                return {"error": f"{type(exc).__name__}: {exc}", "failed_node": name}

        return wrapper

    return deco


def _artifacts(state: SimulationState) -> dict[str, str]:
    return dict(state.get("artifacts") or {})


@_guard("planner")
def planner_node(state: SimulationState) -> dict[str, Any]:
    cfg = state["config"]
    meta: dict[str, Any] = {}
    plan = state.get("plan")
    if plan is not None:
        plan = plan if isinstance(plan, SimulationPlan) else plan_from_dict(plan)
        meta["source"] = "provided plan (--no-llm)"
    else:
        if not state.get("use_llm", True):
            raise ValueError("No plan given and the LLM is disabled (--no-llm needs --plan)")
        from ir_agent.agent.llm import google_invokers, plan_with_llm

        structured, _ = state.get("llm_invokers") or google_invokers(cfg.llm)
        plan, meta = plan_with_llm(state["task"], cfg.llm, structured)
        meta["source"] = "llm"
    run_dir = state.get("run_dir") or RunDir.create(cfg.run.output_root, plan.label)
    setup_logging(cfg.run.log_level, run_dir.path("run.log"))
    run_dir.write_json("plan.json", plan.model_dump())
    log.info("Plan: %s -> %s", plan.description, run_dir.root)
    return {"plan": plan, "plan_meta": meta, "run_dir": run_dir, "artifacts": {"run_dir": str(run_dir.root)}}


@_guard("geometry")
def geometry_node(state: SimulationState) -> dict[str, Any]:
    from ir_agent.pipeline import prepare_system

    ps = prepare_system(state["plan"], state["config"], state["run_dir"])
    return {"prepared": ps}


@_guard("calculation")
def calculation_node(state: SimulationState) -> dict[str, Any]:
    from ir_agent.pipeline import run_vibrations

    res = run_vibrations(state["plan"], state["config"], state["prepared"], state["run_dir"])
    state["run_dir"].write_json("ir_result.json", res.to_dict())
    return {"ir_result": res}


@_guard("critic")
def critic_node(state: SimulationState) -> dict[str, Any]:
    from ir_agent.critic import critique, prepare_retry

    plan, ps, res, cfg = state["plan"], state["prepared"], state["ir_result"], state["config"]
    attempts = int(state.get("critic_attempts") or 0)
    crit = critique(plan, ps, res, cfg)
    entry = {**crit.as_dict(), "attempt": attempts, "actions": []}
    out: dict[str, Any] = {"critic": entry, "retry": False}
    if crit.needs_retry and cfg.critic.enabled and attempts < cfg.critic.max_retries:
        new_cfg, actions = prepare_retry(plan, ps, res, crit, cfg)
        entry["actions"] = actions
        out.update(config=new_cfg, critic_attempts=attempts + 1, retry=True)
    elif crit.issues:
        for i in crit.issues:
            log.warning("critic flag [%s]: %s", i.code, i.message)
    history = [*(state.get("critic_history") or []), entry]
    out["critic_history"] = history
    state["run_dir"].write_json("critic.json", history)
    return out


@_guard("plot_geometry")
def plot_geometry_node(state: SimulationState) -> dict[str, Any]:
    from ir_agent.plotting import plot_structures

    path = state["run_dir"].path("plots", "structures.png")
    plot_structures(state["prepared"].stages, path, title=state["plan"].description)
    return {"artifacts": {**_artifacts(state), "structures_png": str(path)}}


@_guard("spectra")
def spectra_node(state: SimulationState) -> dict[str, Any]:
    from ir_agent.pipeline import make_spectrum
    from ir_agent.plotting import plot_spectrum
    from ir_agent.validation.reference import gas_phase_fundamentals

    plan, res, cfg = state["plan"], state["ir_result"], state["config"]
    spec = make_spectrum(res, cfg)
    refs = None
    if plan.surface is None:
        refs = [f.wavenumber_cm for f in gas_phase_fundamentals(plan.adsorbate) or [] if f.ir_active]
    title = f"IR spectrum: {plan.description}"
    if not res.has_intensities:
        title += " (no dipoles: frequencies only)"
    path = plot_spectrum(spec, state["run_dir"].path("plots", "ir_spectrum.png"), title,
                         label=f"{res.force_method} / {res.dipole_method or 'no dipoles'}", reference_cm=refs,
                         reference_label="NIST fundamentals")
    state["run_dir"].write_json("spectrum.json", {"x_cm": spec.x_cm, "y": spec.y, "units": spec.y_units,
                                                  "sticks_cm": spec.sticks_cm,
                                                  "stick_intensities": spec.stick_intensities})
    return {"spectrum": spec, "artifacts": {**_artifacts(state), "spectrum_png": str(path)}}


@_guard("analysis")
def analysis_node(state: SimulationState) -> dict[str, Any]:
    from ir_agent.analysis import render_report
    from ir_agent.tracking import tracked_run

    plan, res, cfg, rd = state["plan"], state["ir_result"], state["config"], state["run_dir"]
    ps = state["prepared"]
    report, summary = render_report(
        plan_desc=plan.description, calculator=plan.calculator, result=res, atoms=ps.stages["final"],
        adsorbate=plan.adsorbate, metal=plan.surface.metal if plan.surface else None,
        min_cm=cfg.vibrations.min_frequency_cm, scale_factor=cfg.spectrum.scale_factor,
        extra={"opt_steps": ps.opt_steps, "timings_s": {**ps.timings_s, **res.timings_s}},
    )
    report += render_critic(state.get("critic_history") or [])
    if state.get("critic"):
        summary["critic"] = {k: state["critic"][k] for k in ("ok", "n_imag", "lowest_cm", "site", "issues")}
    narrative = ""
    if state.get("use_llm", True):
        try:
            from ir_agent.agent.llm import google_invokers, narrate

            _, text = state.get("llm_invokers") or google_invokers(cfg.llm)
            narrative, model = narrate(report, cfg.llm, text)
            report += f"\n## Analyst summary ({model})\n\n{narrative}\n"
        except Exception as exc:  # the deterministic report is still complete
            log.warning("LLM analyst skipped: %s", exc)
    report_path = rd.path("report.md")
    report_path.write_text(report, encoding="utf-8")
    rd.write_json("summary.json", summary)
    arts = {**_artifacts(state), "report_md": str(report_path)}

    with tracked_run(cfg.tracking, plan.label) as run:
        run.set_tags({"route": res.route, "plan_source": state.get("plan_meta", {}).get("source", "")})
        run.log_params({"calculator": plan.calculator, "adsorbate": plan.adsorbate,
                        "surface": plan.surface.model_dump() if plan.surface else "gas",
                        "dipole_method": res.dipole_method, "delta": cfg.vibrations.delta,
                        "nfree": cfg.vibrations.nfree})
        metrics = {f"time_{k}_s": v for k, v in summary.get("timings_s", {}).items()}
        if "stats" in summary:
            metrics.update({f"freq_{k}": v for k, v in summary["stats"].items()})
        if "band_cm" in summary:
            metrics["band_cm"] = summary["band_cm"]
        run.log_metrics(metrics)
        for p in arts.values():
            if p.endswith((".png", ".md")):
                run.log_artifact(p)
    return {"report_markdown": report, "summary": summary, "narrative": narrative, "artifacts": arts}


def render_critic(history: list[dict[str, Any]]) -> str:
    """Markdown section with the critic's checks and any retries."""
    if not history:
        return ""
    lines = ["", "## Critic checks", ""]
    for h in history:
        head = f"- Attempt {h['attempt'] + 1}: " + ("OK" if h["ok"] else ", ".join(i["code"] for i in h["issues"]))
        lines.append(head)
        lines += [f"  - {i['code']} ({i['action']}): {i['message']}" for i in h["issues"]]
        lines += [f"  - retry: {a}" for a in h.get("actions") or []]
    last = history[-1]
    if last.get("site"):
        lines.append(f"- Relaxed site: {last['site']['label']} (requested {last['site']['requested']})")
    if not last["ok"]:
        lines.append("- Remaining flags are reported, not hidden: treat the affected bands with care.")
    return "\n".join(lines) + "\n"


def failure_node(state: SimulationState) -> dict[str, Any]:
    msg = (f"# Run failed\n\n- Node: `{state.get('failed_node')}`\n- Error: {state.get('error')}\n"
           f"- Task: {state.get('task', '')}\n")
    rd = state.get("run_dir")
    if rd is not None:
        rd.path("report.md").write_text(msg, encoding="utf-8")
    log.error("Workflow stopped in node '%s': %s", state.get("failed_node"), state.get("error"))
    return {"report_markdown": msg}
