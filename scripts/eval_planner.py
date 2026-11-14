"""Planner accuracy table: the live Gemini planner on 20 labelled prompts.

    .venv/Scripts/python scripts/eval_planner.py                     # needs GOOGLE_API_KEY (.env)
    .venv/Scripts/python scripts/eval_planner.py --model gemini-3.6-flash --limit 5
    .venv/Scripts/python scripts/eval_planner.py --model gemini-3.6-flash --no-fallback

Writes docs/planner_eval/<model>.json and .md (field accuracy, exact-plan match, valid-plan rate,
first-try rate, misses). Uses the production planner (prompt, schema, validator feedback and
retries, fallback models). Prompts: examples/planner_eval/prompts.jsonl. Cost: 20-60 small
structured-output calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ir_agent.agent.planner_eval import evaluate, load_cases, render_markdown
from ir_agent.config import load_config

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=Path, default=REPO / "examples" / "planner_eval" / "prompts.jsonl")
    ap.add_argument("--model", help="override llm.model")
    ap.add_argument("--no-fallback", action="store_true",
                    help="use only the configured model (single-model table)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "planner_eval")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    if not os.environ.get("GOOGLE_API_KEY"):
        print("No GOOGLE_API_KEY set: put it in .env (see .env.example) and rerun.\n"
              "Nothing was sent anywhere.", file=sys.stderr)
        return 2
    cfg = load_config()
    llm = cfg.llm.model_copy(update={"model": args.model}) if args.model else cfg.llm
    if args.no_fallback:
        llm = llm.model_copy(update={"fallback_models": []})
    from ir_agent.agent.llm import google_invokers

    structured, _ = google_invokers(llm)
    cases = load_cases(args.cases)[: args.limit]
    result = evaluate(cases, llm, structured)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = llm.model.replace("/", "_") + ("_only" if args.no_fallback else "")
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = render_markdown(result)
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
