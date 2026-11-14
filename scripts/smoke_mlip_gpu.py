"""CUDA smoke test for the open MLIP presets (small jobs, about 1 GB of VRAM, a few minutes).

    .venv-orb/Scripts/python.exe scripts/smoke_mlip_gpu.py orb-mol orb-omat
    .venv-pet/Scripts/python.exe scripts/smoke_mlip_gpu.py pet-omol pet-mad

For each preset: H2O hybrid IR (relax + FD Hessian + xTB dipoles) with the MLIP on CUDA, and for
periodic presets also a CO/Pt(111) 3x3x3 slab FD Hessian (12 displaced geometries). Records wall
time, peak allocated VRAM and the H2O frequencies (to compare with the CPU validation run).
If launched through the optional GPU guard (see ``ir_agent.gpu``), PyTorch is capped to its
reservation (GPU_GUARD_GB). An out-of-memory error is recorded,
not hidden: the preset then has to run on CPU.

Options: preset names; --pet-molecule-ckpt FILE to swap the pet-omol checkpoint (e.g. the S size).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from ir_agent.agent.plan import SimulationPlan
from ir_agent.analysis import match_gas_phase
from ir_agent.calculators.registry import make_force_evaluator
from ir_agent.config import load_config
from ir_agent.gpu import apply_gpu_guard_cap
from ir_agent.pipeline import run_pipeline
from ir_agent.structures import add_molecule_to_slab, build_slab
from ir_agent.vibrations.finite_diff import make_displacements

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("presets", nargs="+")
    ap.add_argument("--pet-molecule-ckpt", default=None)
    ap.add_argument("--max-atoms-per-batch", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    frac = apply_gpu_guard_cap()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    mlip = {"device": "cuda", "max_atoms_per_batch": args.max_atoms_per_batch}
    pet = dict(mlip)
    if args.pet_molecule_ckpt:
        pet["molecule_checkpoint"] = args.pet_molecule_ckpt
    cfg = load_config(overrides={"tracking": {"mlflow": False},
                                 "calculators": {"orb": mlip, "pet": pet}})
    rows = []
    for preset in args.presets:
        row = {"preset": preset, "memory_fraction": frac,
               "pet_checkpoint": pet.get("molecule_checkpoint") if preset == "pet-omol" else None}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            out = run_pipeline(SimulationPlan(adsorbate="H2O", calculator=preset), cfg)
            res = out["result"]
            row["h2o_wall_s"] = time.perf_counter() - t0
            row["h2o_timings_s"] = res.timings_s
            row["h2o_freqs_cm"] = [round(m.calc_cm, 1) for m in match_gas_phase("H2O", res)]
            row["h2o_peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
            if preset in ("orb-omat", "pet-mad"):
                torch.cuda.reset_peak_memory_stats()
                s = add_molecule_to_slab(build_slab("Pt", "fcc", "111", (3, 3, 3)), "CO", 1.9)
                ev = make_force_evaluator(preset, cfg, atoms=s.atoms, system_kind="surface")
                _, images = make_displacements(s.atoms, s.adsorbate_indices, 0.01, 2)
                t1 = time.perf_counter()
                ev(images)
                row["slab_fd_wall_s"] = time.perf_counter() - t1
                row["slab_n_images"] = len(images)
                row["slab_peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
            row["status"] = "ok"
        except torch.cuda.OutOfMemoryError as exc:
            row.update(status="out of memory under the guard cap", error=str(exc)[:300],
                       wall_s=time.perf_counter() - t0)
        except Exception as exc:  # recorded, the next preset still runs
            row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500],
                       trace=traceback.format_exc()[-1500:], wall_s=time.perf_counter() - t0)
        print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
        rows.append(row)
    out_path = Path(args.out or ROOT / "runs" / f"mlip_gpu_smoke_{datetime.now():%Y%m%d-%H%M%S}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "guard_gb": float(__import__("os").environ.get("GPU_GUARD_GB", "0") or 0)}
    out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
