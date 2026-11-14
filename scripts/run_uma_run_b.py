"""Run B: UMA-S 1.2.1 CO site x task x coverage on Pt(111).

    python scripts/run_uma_run_b.py            # 6 resumable validation runs on CUDA
    python scripts/run_uma_run_b.py --dry-run  # print the commands only

Stdlib only (GPU jobs are wrapped in the optional guard, see ``ir_agent.gpu``). Design:

* task {oc20, omat} (periodic slab tasks; each with its own same-task gas references),
* coverage {3x3, 4x4} Pt(111) cell, 3 layers (1/9 vs 1/16 ML),
* CO on {ontop, bridge, fcc} (the CO/Pt(111) site question; GGA-level DFT tends to over-bind the
  hollow site) and NH3 ontop (NH3 binds atop; other sites would only relax back),
* numerical check: finite-difference step delta 0.02 A next to the default 0.01 A for the ontop
  cases on 3x3, both tasks.

Every case records the relaxed site (critic), so site hops are visible. That gives 6 points and
20 adsorbate cases plus 2 gas molecules per point. Outputs: runs/uma_run_b/<point>/validation
(resumable with the same command) and runs/uma_run_b/summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_agent.gpu import guarded  # noqa: E402  (stdlib-only module)

PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
OUT = ROOT / "runs" / "uma_run_b"

SITES = "CO/Pt/111,CO/Pt/111/bridge,CO/Pt/111/fcc,NH3/Pt/111"
ONTOP = "CO/Pt/111,NH3/Pt/111"
#: (task preset, lateral cell, FD step, surfaces)
POINTS = [
    ("uma-oc20", 3, 0.01, SITES),
    ("uma-oc20", 4, 0.01, SITES),
    ("uma-omat", 3, 0.01, SITES),
    ("uma-omat", 4, 0.01, SITES),
    ("uma-oc20", 3, 0.02, ONTOP),  # step-size check
    ("uma-omat", 3, 0.02, ONTOP),
]


def point_config(cell: int, delta: float) -> str:
    return (
        "vibrations:\n"
        f"  delta: {delta}\n"
        "geometry:\n"
        f"  slab_size: [{cell}, {cell}, 3]\n"
        "tracking:\n  mlflow: true\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gb", default="3.0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for preset, cell, delta, surfaces in POINTS:
        name = f"{preset}_c{cell}x{cell}_d{delta}"
        pdir = OUT / name
        pdir.mkdir(exist_ok=True)
        ov = pdir / "overrides.yaml"  # deep-merged over configs/default.yaml by the CLI
        ov.write_text(point_config(cell, delta), encoding="utf-8")
        resume = pdir / "validation"
        cmd = [str(PY), "-m", "ir_agent", "--overrides", str(ov), "validate", "--calculators", preset,
               "--molecules", "CO,NH3", "--surfaces", surfaces, "--device", "cuda", "--resume", str(resume)]
        full = guarded(cmd, args.gb, f"UMA Run B {name}")
        print(" ".join(full), flush=True)
        if args.dry_run:
            continue
        for sub in ("structures", "calc", "plots"):
            (resume / sub).mkdir(parents=True, exist_ok=True)
        rc = subprocess.call(full, cwd=ROOT)
        row = {"point": name, "preset": preset, "cell": cell, "delta": delta, "exit": rc}
        vj = resume / "validation.json"
        if vj.exists():
            d = json.loads(vj.read_text(encoding="utf-8"))
            row["bands"] = {
                r["system"]: {"band_cm": r.get("band_cm"), "calc_shift_cm": r.get("calc_shift_cm"),
                              "site": (r.get("site") or {}).get("label"), "flags": r.get("flags"),
                              "error": r.get("error")}
                for r in d["adsorbates"]
            }
        summary.append(row)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
