"""Run C: adsorbate systems from semiconductor processing.

    python scripts/run_c.py --cpu                 # GFN1-xTB part (CPU, 4 threads, needs >= 8 GB free RAM)
    python scripts/run_c.py --gpu                 # UMA part (uma-omol, uma-oc20, uma-omat) on CUDA
    python scripts/run_c.py --gpu --dry-run       # print the command (wrapped in the optional GPU guard)

Systems: CO atop Cu(111) and Ru(0001) (shift vs same-task gas CO), H/Si(100)-2x1 monohydride,
NH2+H/Si(100)-2x1 and an isolated silanol (absolute wavenumbers vs experiment). Each part writes
its own run directory (runs/<stamp>_run_c) with run_c.json/.md and validation.json/.md; rerun with
--resume <dir> to continue an interrupted part.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_uma_run_a import MIN_FREE_GB, free_ram_gb, guarded  # noqa: E402  (stdlib helpers)

PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
CPU_CALCS = ["xtb-gfn1"]
GPU_CALCS = ["uma-omol", "uma-oc20", "uma-omat"]


def inner(calculators: list[str], device: str | None, resume: str | None, parts: tuple[str, ...]) -> int:
    from ir_agent.config import load_config
    from ir_agent.gpu import apply_gpu_guard_cap
    from ir_agent.validation.run_c import run_c

    apply_gpu_guard_cap()  # no-op unless launched through the optional GPU guard
    ov: dict = {}
    if device:
        ov = {"calculators": {"uma": {"device": device}}}
    cfg = load_config(overrides=ov)
    out = run_c(cfg, calculators, resume_dir=resume, parts=parts)
    print(f"Run C written to {out['dir']}")
    errors = [r for r in out.get("semis", []) if r.get("error")]
    errors += [r for r in out.get("metal", {}).get("adsorbates", []) if r.get("error")]
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu", action="store_true", help="GFN1-xTB references on the CPU")
    g.add_argument("--gpu", action="store_true", help="UMA presets on CUDA (optional GPU guard: env GPU_GUARD)")
    g.add_argument("--inner", nargs="+", metavar="CALC", help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--resume", default=None, help="continue in an existing runs/<stamp>_run_c dir")
    ap.add_argument("--gb", default="3.0")
    ap.add_argument("--parts", default="metal,semis", help="metal, semis or both (default)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.inner:
        return inner(args.inner, args.device, args.resume, tuple(args.parts.split(",")))
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    resume = ["--resume", args.resume] if args.resume else []
    resume += ["--parts", args.parts]
    if args.cpu:
        cmd = [str(PY), str(Path(__file__).resolve()), "--inner", *CPU_CALCS, *resume]
        if args.dry_run:
            print("CPU:", " ".join(cmd), "(OMP_NUM_THREADS=4)")
            return 0
        free = free_ram_gb()
        if free is not None and free < MIN_FREE_GB:
            print(f"Not starting: {free:.1f} GB RAM free (< {MIN_FREE_GB:g} GB).", file=sys.stderr)
            return 3
        return subprocess.call(cmd, cwd=ROOT, env=env)
    cmd = guarded([str(PY), str(Path(__file__).resolve()), "--inner", *GPU_CALCS, "--device", "cuda", *resume],
                  args.gb, "Run C: UMA on Cu/Ru/Si/SiO2 systems")
    if args.dry_run:
        print("GPU:", " ".join(cmd))
        return 0
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
