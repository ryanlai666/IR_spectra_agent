"""Command-line entry point (replaces v0.1 ``test_endtoend.py``).

Examples::

    ir-agent run "IR spectrum of CO on Pt(111) with UMA"           # LLM planner (GOOGLE_API_KEY)
    ir-agent run --no-llm --plan examples/plans/h2o_gas_xtb.json    # deterministic, no API key
    ir-agent calculators                                            # what is installed here
    ir-agent validate --calculators emt,xtb,dxtb                    # NIST validation table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ir_agent.config import load_config
from ir_agent.runtime import get_logger, setup_logging

log = get_logger("cli")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass


def cmd_run(args: argparse.Namespace) -> int:
    from ir_agent.agent.graph import build_graph
    from ir_agent.agent.plan import load_plan

    cfg = load_config(args.config, overrides=_overrides(args))
    setup_logging(cfg.run.log_level)
    state: dict = {"config": cfg, "use_llm": not args.no_llm, "task": args.task or ""}
    if args.plan:
        plan = load_plan(args.plan)
        if args.calculator:
            plan = plan.model_copy(update={"calculator": args.calculator})
            plan = type(plan).model_validate(plan.model_dump())
        state["plan"] = plan
        state["task"] = args.task or plan.task or plan.description
    elif args.no_llm:
        log.error("--no-llm requires --plan PLAN.json")
        return 2
    elif not args.task:
        state["task"] = input("Describe the IR simulation (e.g. 'CO on Pt(111) with xtb'): ").strip()

    app = build_graph()
    final = app.invoke(state)
    print(final.get("report_markdown", "(no report)"))
    arts = final.get("artifacts", {})
    if arts:
        print("\nArtifacts:")
        for k, v in arts.items():
            print(f"  {k}: {v}")
    return 1 if final.get("error") else 0


def cmd_calculators(args: argparse.Namespace) -> int:
    from ir_agent.calculators.registry import list_presets

    cfg = load_config(args.config)
    rows = list_presets(cfg)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'preset':14s} {'ok':3s} {'PBC':4s} {'dipole':6s} {'GPU':4s} {'platforms':38s} status")
    for r in rows:
        print(f"{r['name']:14s} {'yes' if r['available'] else 'no':3s} {'y' if r['periodic'] else 'n':4s} "
              f"{'y' if r['native_dipole'] else 'n':6s} {'y' if r['gpu'] else 'n':4s} "
              f"{r['platforms'][:38]:38s} {r['status']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from ir_agent.validation.run import run_validation

    cfg = load_config(args.config, overrides=_overrides(args))
    setup_logging(cfg.run.log_level)
    calcs = [c.strip() for c in args.calculators.split(",") if c.strip()]
    mols = [m.strip() for m in args.molecules.split(",") if m.strip()]
    surfaces = [] if args.no_surfaces else [s.strip() for s in args.surfaces.split(",") if s.strip()]
    out = run_validation(cfg, calcs, mols, surfaces, skip_unavailable=not args.strict,
                         resume_dir=args.resume)
    print(out["markdown"])
    print(f"\nWritten to {out['dir']}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from ir_agent.agent.graph import build_graph, mermaid, save_graph_png

    app = build_graph()
    print(mermaid(app))
    if args.png:
        p = save_graph_png(app, Path(args.png))
        print(f"PNG: {p or 'failed (needs network for mermaid.ink)'}")
    return 0


def _overrides(args: argparse.Namespace) -> dict:
    ov: dict = {}
    if getattr(args, "overrides", None):
        import yaml

        with open(args.overrides, encoding="utf-8") as fh:
            ov = yaml.safe_load(fh) or {}
    if getattr(args, "output_root", None):
        ov.setdefault("run", {})["output_root"] = args.output_root
    if getattr(args, "device", None):  # applies to every MLIP preset (uma, orb-*, pet-*)
        for mlip in ("uma", "orb", "pet"):
            ov.setdefault("calculators", {}).setdefault(mlip, {})["device"] = args.device
    if getattr(args, "no_mlflow", False):
        ov.setdefault("tracking", {})["mlflow"] = False
    return ov


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ir-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--overrides", default=None, help="extra YAML deep-merged over --config (e.g. sweeps)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="plan and run one IR simulation")
    r.add_argument("task", nargs="?", help="natural-language request (LLM planner)")
    r.add_argument("--plan", help="JSON plan file (skips the LLM planner)")
    r.add_argument("--no-llm", action="store_true", help="deterministic mode: no LLM calls at all")
    r.add_argument("--calculator", help="override the plan's calculator preset")
    r.add_argument("--output-root", help="override run.output_root")
    r.add_argument("--device", help="MLIP device for uma/orb/pet presets (cuda/cpu)")
    r.add_argument("--no-mlflow", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calculators", help="list calculator presets and availability")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_calculators)

    v = sub.add_parser("validate", help="gas-phase NIST + adsorbate trend validation")
    v.add_argument("--calculators", default="xtb,xtb-gfn1,uma-omol,uma-oc20,uma-omat",
                   help="presets; uma-<task> pins one UMA task for gas reference and adsorbate")
    v.add_argument("--molecules", default="H2O,CO,NH3,CO2")
    v.add_argument("--surfaces", default="CO/Pt/111,NH3/Pt/111,CO/Au/111,NH3/Au/111",
                   help="MOL/METAL/FACET[/SITE] list; SITE = ontop (default), bridge, fcc, hcp")
    v.add_argument("--no-surfaces", action="store_true")
    v.add_argument("--strict", action="store_true", help="fail instead of skipping unavailable calculators")
    v.add_argument("--resume", help="existing validation run dir: reuse finished sub-runs (checkpoint/resume)")
    v.add_argument("--output-root")
    v.add_argument("--device")
    v.add_argument("--no-mlflow", action="store_true")
    v.set_defaults(func=cmd_validate)

    g = sub.add_parser("graph", help="print the LangGraph (mermaid)")
    g.add_argument("--png", help="also write a PNG (uses mermaid.ink)")
    g.set_defaults(func=cmd_graph)
    return p


def main(argv: list[str] | None = None) -> int:
    _load_env()
    from ir_agent.gpu import apply_gpu_guard_cap

    apply_gpu_guard_cap()  # no-op unless launched through the shared gpu_guard
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":
    sys.exit(main())
