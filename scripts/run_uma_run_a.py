"""Run A: UMA-S 1.2.1 validation (gas molecules + CO/NH3 on Pt/Au), one command.

    python scripts/run_uma_run_a.py                 # GPU job, then CPU xTB reference if missing
    python scripts/run_uma_run_a.py --dry-run       # print the commands only

Stdlib only. Steps:

1. Verify data/checkpoints/uma-s-1p2p1.pt (UMA-S 1.2.1) against its published MD5 (fail loudly otherwise).
2. GPU, one command (wrapped in the optional GPU guard, see ``ir_agent.gpu``, with --gb 3.0):
       validate --calculators uma-omol,uma-oc20,uma-omat --device cuda
   * uma-omol: gas molecules + Pt10/Au10 cluster models (omol task for both),
   * uma-oc20, uma-omat: gas molecules + periodic 3x3x3 slabs (one task for both),
   so every adsorption shift is referenced to a gas run of the *same* task.
   IR intensities: GFN2-xTB fragment dipoles (CPU, inside the job; small).
3. CPU, after the GPU job: GFN1-xTB references (xtb executable, ``xtb --hess``) with
   at most 4 threads and only if >= 8 GB RAM is free. Skipped when
   docs/validation/validation_xtb.json (same matrix, current code) already exists, unless
   --cpu-ref run.
4. Write runs/uma_run_a_<stamp>.md (status per task, wall time, peak VRAM, run dirs).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_agent.gpu import guarded  # noqa: E402  (stdlib-only module)

CKPT = ROOT / "data" / "checkpoints" / "uma-s-1p2p1.pt"
MD5 = "3497615fd30a24c5b35cd3b41a682e6e"
GB = "3.0"
PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
UMA_PRESETS = ["uma-omol", "uma-oc20", "uma-omat"]
MOLECULES = "H2O,CO,NH3,CO2"
SURFACES = "CO/Pt/111,NH3/Pt/111,CO/Au/111,NH3/Au/111"
GPU_CMD = [str(PY), "-m", "ir_agent", "validate", "--calculators", ",".join(UMA_PRESETS),
           "--molecules", MOLECULES, "--surfaces", SURFACES, "--device", "cuda"]
CPU_CMD = [str(PY), "-m", "ir_agent", "validate", "--calculators", "xtb-gfn1",
           "--molecules", MOLECULES, "--surfaces", SURFACES]
XTB_REF = ROOT / "docs" / "validation" / "validation_xtb.json"
MIN_FREE_GB = 8.0


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_ram_gb() -> float | None:
    """Available physical memory (GB); None if it cannot be read."""
    try:
        if sys.platform.startswith("win"):
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1024**3
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024**2
    except Exception:
        return None
    return None


def newest_validation_dir(after: float) -> Path | None:
    runs = [p for p in (ROOT / "runs").glob("*_validation") if p.stat().st_mtime >= after]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def run_logged(cmd: list[str], log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)


def summarize_uma(run_dir: Path | None) -> tuple[list[str], float | None, int, int]:
    """(markdown lines, peak VRAM MB, number of UMA cases, number of failed cases)."""
    lines, peaks, n_all, n_bad = [], [], 0, 0
    if not run_dir or not (run_dir / "validation.json").exists():
        return ["- no validation.json found"], None, 0, 0
    d = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    for preset in UMA_PRESETS:
        rows = [r for r in d["gas"] + d["adsorbates"] if r["calculator"] == preset]
        n_err = sum(1 for r in rows if r.get("error"))
        n_flag = sum(1 for r in rows if r.get("flags"))
        n_all, n_bad = n_all + len(rows), n_bad + n_err
        peaks += [r["gpu_peak_mb"] for r in rows if r.get("gpu_peak_mb")]
        shifts = ", ".join(f"{r['system']} {r['calc_shift_cm']:+.0f}" for r in d["adsorbates"]
                           if r["calculator"] == preset and r.get("calc_shift_cm") is not None)
        lines.append(f"- `{preset}`: {len(rows)} cases, {n_err} errors, {n_flag} flagged by the critic; "
                     f"shifts (cm^-1): {shifts or 'n/a'}")
    if d.get("skipped"):
        lines.append(f"- skipped: {d['skipped']}")
    return lines, (max(peaks) if peaks else None), n_all, n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-ref", choices=["auto", "run", "skip"], default="auto",
                    help="GFN1-xTB CPU reference after the GPU job (auto: only if validation_xtb.json is missing)")
    args = ap.parse_args()
    gpu_full = guarded(GPU_CMD, GB, "UMA Run A: uma-omol/oc20/omat validation on CUDA")
    if args.dry_run:
        print("GPU:", " ".join(gpu_full))
        print("CPU:", " ".join(CPU_CMD), "(OMP_NUM_THREADS=4)")
        return 0
    if not CKPT.is_file():
        print(f"ERROR: {CKPT} not found (expected uma-s-1p2p1.pt, MD5 {MD5}).", file=sys.stderr)
        return 2
    got = md5sum(CKPT)
    if got != MD5:
        print(f"ERROR: MD5 mismatch for {CKPT}: got {got}, expected {MD5}. Not running.", file=sys.stderr)
        return 3
    print(f"MD5 OK ({got})")
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}

    t0 = time.time()
    print("launch:", " ".join(gpu_full), flush=True)
    rc = run_logged(gpu_full, ROOT / "runs" / f"uma_run_a_{stamp}.log", env)
    gpu_wall = time.time() - t0
    gpu_dir = newest_validation_dir(t0)
    uma_lines, peak, n_cases, n_failed = summarize_uma(gpu_dir)

    cpu_note = "skipped (--cpu-ref skip)"
    if args.cpu_ref == "run" or (args.cpu_ref == "auto" and not XTB_REF.exists()):
        free = free_ram_gb()
        if free is not None and free < MIN_FREE_GB:
            cpu_note = f"not run: only {free:.1f} GB RAM free (< {MIN_FREE_GB:g} GB); rerun with --cpu-ref run later"
        else:
            t1 = time.time()
            crc = run_logged(CPU_CMD, ROOT / "runs" / f"uma_run_a_cpu_ref_{stamp}.log", env)
            cdir = newest_validation_dir(t1)
            cpu_note = (f"exit {crc}, {(time.time() - t1) / 60:.1f} min, run dir "
                        f"`{cdir.relative_to(ROOT) if cdir else 'n/a'}`")
    elif args.cpu_ref == "auto":
        cpu_note = f"reused `{XTB_REF.relative_to(ROOT)}` (same matrix, xtb --hess route)"

    status = ("done" if rc == 0 and n_cases and not n_failed
              else "partial" if rc == 0 else f"failed (exit {rc})")
    md = [
        "# UMA Run A summary",
        "",
        f"- Status: **{status}**",
        f"- Finished: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- GPU command: `{' '.join(GPU_CMD)}`" + (f" (GPU guard, --gb {GB})" if gpu_full != GPU_CMD else ""),
        f"- Checkpoint: `data/checkpoints/uma-s-1p2p1.pt` (UMA-S 1.2.1), MD5 {got} (matches)",
        f"- GPU wall time: {gpu_wall / 60:.1f} min",
        f"- Peak VRAM (torch max_memory_allocated): {'n/a' if peak is None else f'{peak / 1024:.2f} GiB'}",
        f"- GPU run dir: `{gpu_dir.relative_to(ROOT) if gpu_dir else 'n/a'}`",
        "",
        "Per task (gas reference and adsorbate use the same task):",
        *uma_lines,
        "",
        f"- GFN1-xTB CPU reference (after the GPU job): {cpu_note}",
    ]
    (ROOT / "runs" / f"uma_run_a_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return rc


if __name__ == "__main__":
    sys.exit(main())
