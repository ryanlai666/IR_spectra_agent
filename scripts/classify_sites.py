"""Classify the adsorption site of every adsorbate case in one or more validation run directories.

Writes docs/validation/adsorbate_sites.json (replaces the earlier hand-made file) and
docs/validation/convergence.json (final fmax of every gas and adsorbate sub-run, read from
``calc/final_opt.log``; rows written before the critic existed carry no convergence field):

    .venv/Scripts/python scripts/classify_sites.py runs/<stamp>_validation [more roots ...]

Each ``ads_*`` sub-run contributes its relaxed ``structures/final.xyz``; the adsorbate is every
non-metal atom. Later roots override earlier ones for the same (system, calculator) key, so pass
the roots oldest first. The rule is in ``ir_agent.sites.classify_site``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ase.io import read

from ir_agent.sites import classify_site, site_changed

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "validation" / "adsorbate_sites.json"
CONV = REPO / "docs" / "validation" / "convergence.json"
FMAX_FINAL = 0.01  # configs/default.yaml optimization.fmax_final
NAME = re.compile(r"ads_(?P<mol>[A-Za-z0-9]+)_(?P<metal>[A-Z][a-z]?)(?P<facet>\d+)(?:_(?P<site>ontop|bridge|fcc|hcp))?_(?P<calc>.+)$")
OPTIMIZERS = ("BFGS", "FIRE", "LBFGS", "BFGSLineSearch")


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def final_fmax(sub: Path) -> float | None:
    """Last fmax in calc/final_opt.log (ASE optimizer log: name step time energy fmax)."""
    log = sub / "calc" / "final_opt.log"
    if not log.is_file():
        return None
    rows = [ln.split() for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()]
    rows = [r for r in rows if len(r) >= 5 and r[0].rstrip(":") in OPTIMIZERS]
    return float(rows[-1][-1]) if rows else None


def convergence_root(root: Path) -> dict[str, dict]:
    out = {}
    for sub in sorted([*root.glob("gas_*"), *root.glob("ads_*")]):
        f = final_fmax(sub)
        if f is not None:
            out[rel(sub)] = {"final_fmax": f, "converged": f <= FMAX_FINAL + 1e-9}
    return out


def classify_root(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sub in sorted(root.glob("ads_*")):
        final = sub / "structures" / "final.xyz"
        m = NAME.match(sub.name)
        if not final.is_file() or not m:
            continue
        atoms = read(final)
        info = classify_site(atoms)
        requested = m["site"] or "ontop"
        system = f"{m['mol']}/{m['metal']}({m['facet']})" + (f"/{m['site']}" if m["site"] else "")
        out[f"{system}|{m['calc']}"] = {
            **info.as_dict(), "requested": requested, "site_changed": site_changed(requested, info),
            "source": rel(sub), "final_fmax": final_fmax(sub),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path, help="validation run directories (oldest first)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--conv-out", type=Path, default=CONV)
    args = ap.parse_args()
    sites: dict[str, dict] = {}
    conv: dict[str, dict] = {}
    for root in args.roots:
        root = root if root.is_absolute() else REPO / root
        sites.update(classify_root(root))
        conv.update(convergence_root(root))
    args.out.write_text(json.dumps(sites, indent=1) + "\n", encoding="utf-8")
    args.conv_out.write_text(json.dumps(conv, indent=1) + "\n", encoding="utf-8")
    for k, v in sites.items():
        flag = "  <-- site changed" if v["site_changed"] else ""
        print(f"{k:28s} {v['label']}{flag}")
    for k, v in conv.items():
        if not v["converged"]:
            print(f"not converged: {k} (final fmax {v['final_fmax']:.3f} eV/A)")
    print(f"wrote {args.out} ({len(sites)} cases) and {args.conv_out}")


if __name__ == "__main__":
    main()
