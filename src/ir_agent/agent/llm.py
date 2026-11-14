"""Gemini-backed planner and analyst with validation-driven retries and model fallback.

The LLM is optional: ``ir-agent run --no-llm --plan plan.json`` never imports it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from ir_agent.agent.plan import PlannerOutput, SimulationPlan
from ir_agent.config import LLMConfig
from ir_agent.runtime import get_logger

log = get_logger("llm")

Messages = Sequence[tuple[str, str]]
#: (model_name, messages) -> PlannerOutput | dict
StructuredInvoke = Callable[[str, Messages], Any]
#: (model_name, prompt) -> str
TextInvoke = Callable[[str, str], str]


class LLMUnavailableError(RuntimeError):
    pass


class PlanningError(RuntimeError):
    pass


PLANNER_SYSTEM_PROMPT = """\
You are the planning agent of an automated computational-chemistry workflow that computes IR
spectra of gas-phase molecules and of molecules adsorbed on metal surfaces.
Extract the simulation parameters from the user's request.

Calculator presets (map synonyms):
- 'emt': fast, cheap, mock, test -> EMT toy potential
- 'uma': UMA, fairchem, Meta, machine-learning potential, MLIP, GPU (task chosen per system)
- 'uma-omol' / 'uma-oc20' / 'uma-omat': UMA with one named task (omol, oc20/OC20, omat/OMat)
- 'orb-mol': OrbMol, Orb molecular model; 'orb-omat': Orb v3, Orb OMat (periodic materials)
- 'pet-omol': PET-OMol, UPET molecular model; 'pet-mad': PET-MAD
- 'xtb': xtb, GFN2, GFN2-xTB, semi-empirical, tight-binding
- 'tblite': tblite
- 'xtb-gfn1': GFN1-xTB with the xtb program (robust for metal clusters)
- 'dxtb': dxtb, GFN1 in PyTorch
- 'dftb+': DFTB, DFTB+
- 'gpaw': GPAW, PAW
- 'nwchem': NWChem
- 'vasp': VASP, DFT, PBE, default/medium accuracy; 'vasp-accurate': high accuracy
Rules:
- If no calculator is named use 'emt'. If no molecule is named use 'H2O'.
- adsorbate must be an ASE g2 name (H2O, CO, NH3, CO2, CH4, N2, NO, CH3OH, HCN, ...).
- If the request is about an isolated/gas-phase molecule (no surface, "not on a metal"),
  set surface_metal to 'None'. Otherwise give the element symbol (gold -> 'Au', platinum -> 'Pt').
- surface_structure is the metal's bulk lattice (Au, Pt, Cu, Ag, Pd, Ni, Al: 'fcc'; Fe, W, Mo:
  'bcc'; Ru, Co, Ti, Zn, Mg: 'hcp'). Default facet: '111' for fcc, '110' for bcc, '0001' for hcp.
- dipole_method: 'auto' unless the user explicitly names one of native/xtb/tblite/dxtb/none.
- surface_site: 'ontop' unless the request names 'bridge', 'fcc' (fcc hollow), 'hcp' (hcp hollow);
  'hollow' alone means 'fcc'.
- surface_model: 'cluster' or 'slab' only if the user asks for one, else 'auto'.
"""


def _model_missing(exc: Exception) -> bool:
    s = str(exc)
    return any(k in s for k in ("404", "NOT_FOUND", "not found", "is not supported", "deprecated"))


def google_invokers(cfg: LLMConfig) -> tuple[StructuredInvoke, TextInvoke]:
    """Build Gemini invokers. Raises LLMUnavailableError with a clear message if not usable."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise LLMUnavailableError(
            "GOOGLE_API_KEY is not set. Put it in .env (see .env.example) or run with "
            "`--no-llm --plan plan.json`."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise LLMUnavailableError(
            "langchain-google-genai is not installed: uv pip install langchain-google-genai"
        ) from exc

    def _chat(model: str):
        return ChatGoogleGenerativeAI(
            model=model, temperature=cfg.temperature, timeout=cfg.timeout_s, max_retries=1
        )

    def structured(model: str, messages: Messages) -> Any:
        return _chat(model).with_structured_output(PlannerOutput).invoke(list(messages))

    def text(model: str, prompt: str) -> str:
        out = _chat(model).invoke(prompt)
        content = getattr(out, "content", out)
        if isinstance(content, list):  # multi-part responses
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content)

    return structured, text


def _short(exc: Exception, limit: int = 600) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc'])) or 'plan'}: {e['msg']}" for e in exc.errors()]
        s = "; ".join(parts)
    else:
        s = str(exc)
    return s[:limit]


def plan_with_llm(
    task: str, cfg: LLMConfig, invoke: StructuredInvoke
) -> tuple[SimulationPlan, dict[str, Any]]:
    """Ask the LLM for a plan; validate it; feed validation errors back and retry.

    Tries ``cfg.model`` then ``cfg.fallback_models``; a model that does not exist (retired)
    is skipped immediately. Returns the plan and metadata (model used, attempts, errors).
    """
    models = [cfg.model, *[m for m in cfg.fallback_models if m != cfg.model]]
    errors: list[str] = []
    for model in models:
        feedback: str | None = None
        for attempt in range(1, cfg.max_retries + 1):
            messages: list[tuple[str, str]] = [
                ("system", PLANNER_SYSTEM_PROMPT),
                ("human", f"User task: {task}"),
            ]
            if feedback:
                messages.append(
                    ("human", f"Your previous answer was rejected by the validator: {feedback}\n"
                              "Return a corrected answer that satisfies the rules.")
                )
            try:
                raw = invoke(model, messages)
            except Exception as exc:  # network / API errors
                errors.append(f"{model}#{attempt}: {_short(exc)}")
                if _model_missing(exc):
                    log.warning("LLM model %s unavailable (%s); trying fallback", model, _short(exc, 120))
                    break
                continue
            try:
                if raw is None:
                    raise ValueError("empty structured output")
                out = raw if isinstance(raw, PlannerOutput) else PlannerOutput.model_validate(raw)
                plan = out.to_plan(task)
            except (ValidationError, ValueError) as exc:
                feedback = _short(exc)
                errors.append(f"{model}#{attempt}: invalid plan: {feedback}")
                log.info("Plan rejected (attempt %d): %s", attempt, feedback)
                continue
            return plan, {"model": model, "attempts": attempt, "errors": errors}
    raise PlanningError("LLM planning failed:\n  " + "\n  ".join(errors))


ANALYST_PROMPT = """\
You are a computational spectroscopist. Write one concise paragraph (<= 150 words) for the
report below: which bands dominate, how they compare to the experimental reference values given,
and the main caveats of the method (harmonic approximation, level of theory, intensity model).
Do not invent numbers that are not in the report.

{report}
"""


def narrate(report_markdown: str, cfg: LLMConfig, invoke: TextInvoke) -> tuple[str, str]:
    """Return (paragraph, model) or raise LLMUnavailableError if all models fail."""
    last: Exception | None = None
    for model in [cfg.model, *cfg.fallback_models]:
        try:
            return invoke(model, ANALYST_PROMPT.format(report=report_markdown)).strip(), model
        except Exception as exc:  # pragma: no cover - network
            last = exc
            log.warning("Analyst model %s failed: %s", model, _short(exc, 200))
    raise LLMUnavailableError(f"All analyst models failed: {last}")
