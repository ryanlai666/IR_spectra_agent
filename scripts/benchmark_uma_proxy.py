"""GPU benchmark of the batched finite-difference Hessian with a *random-weight* UMA-S proxy.

Why: the real UMA checkpoints are gated (facebook/UMA), but throughput and peak VRAM depend on
the architecture, not on the weights. This builds fairchem's
eSCN-MD MoLE backbone with the UMA-S hyperparameters from the UMA paper (arXiv:2506.23971,
appendix table: 4 layers, 128 sphere channels, lmax=2, mmax=2, 32 experts, 64 radial basis
functions, 6 A cutoff, 300 max neighbors; hidden/edge channels not stated -> fairchem default 128)
and times the same batched force evaluation the 'uma' preset uses.

The numbers are a *proxy*: frequencies from random weights are meaningless. With the real
checkpoint use:  pytest -m "gpu and uma" tests/test_uma.py

Usage:
  python scripts/benchmark_uma_proxy.py
On a shared GPU, prefix the command with the optional guard (see ``ir_agent.gpu``) and a memory
reservation, e.g. ``python $GPU_GUARD run --agent ir-agent --gb 3.5 -- python scripts/benchmark_uma_proxy.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ase.build import molecule

from ir_agent.structures import add_molecule_to_slab, build_slab
from ir_agent.vibrations.finite_diff import make_displacements

UMA_S = dict(
    num_experts=32, sphere_channels=128, lmax=2, mmax=2, num_layers=4, num_distance_basis=64,
    cutoff=6.0, max_neighbors=300, hidden_channels=128, edge_channels=128,
)


def build_proxy(device: str):
    from fairchem.core.models.uma.escn_md import MLP_EFS_Head
    from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

    datasets = ["omol", "oc20", "omat", "odac", "omc"]
    backbone = eSCNMDMoeBackbone(
        **UMA_S, otf_graph=True, regress_forces=True, direct_forces=False,
        dataset_mapping={d: d for d in datasets}, use_dataset_embedding=True,
        use_composition_embedding=True,
    )
    head = MLP_EFS_Head(backbone)
    backbone.to(device).eval()
    head.to(device).eval()
    n_total = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())
    return backbone, head, n_total


def to_batch(images, task: str, device: str):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    data = []
    for img in images:
        a = img.copy()
        if task == "omol":
            a.info.update(charge=0, spin=1)
        data.append(AtomicData.from_ase(a, task_name=task, r_data_keys=["spin", "charge"], radius=6.0))
    return atomicdata_list_to_batch(data).to(device)


def time_system(name, atoms, idx, task, backbone, head, device, batch_sizes, repeats=2, max_atoms=180):
    _, images = make_displacements(atoms, idx, 0.01, 2)
    rows = []
    for bs in batch_sizes:
        if bs > 1 and bs * len(atoms) > max_atoms:  # VRAM guard (shared 8 GB GPU, < 3 GB budget)
            continue
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for rep in range(repeats + 1):  # first pass = warm-up
            t0 = time.perf_counter()
            n_pass = 0
            for s in range(0, len(images), bs):
                batch = to_batch(images[s : s + bs], task, device)
                emb = backbone(batch)
                out = head(batch, emb)
                f = out["forces"]["forces"] if isinstance(out["forces"], dict) else out["forces"]
                _ = f.detach().cpu()
                n_pass += 1
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if rep > 0:
                times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
        rows.append({
            "system": name, "n_atoms": len(atoms), "n_displaced_atoms": len(idx),
            "n_images": len(images), "batch_size": bs, "forward_passes": n_pass,
            "hessian_wall_s": float(np.mean(times)), "per_image_ms": 1e3 * float(np.mean(times)) / len(images),
            "peak_vram_mb": peak,
        })
        print(json.dumps(rows[-1]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-sizes", default="1,6,12,24")
    ap.add_argument("--max-atoms-per-batch", type=int, default=180)
    ap.add_argument("--quick", action="store_true", help="H2O only, batch sizes 1 and 6")
    args = ap.parse_args()
    torch.manual_seed(0)
    from ir_agent.gpu import apply_gpu_guard_cap, warn_if_unguarded

    frac = apply_gpu_guard_cap()
    warn_if_unguarded(args.device)
    bss = [int(b) for b in args.batch_sizes.split(",")]

    t0 = time.perf_counter()
    backbone, head, n_params = build_proxy(args.device)
    print(f"UMA-S proxy: {n_params/1e6:.1f}M total parameters (paper: ~150M total / 6M active)")

    systems = []
    h2o = molecule("H2O")
    h2o.center(vacuum=6.0)
    systems.append(("H2O gas (omol)", h2o, list(range(3)), "omol"))
    nh3 = molecule("NH3")
    nh3.center(vacuum=6.0)
    systems.append(("NH3 gas (omol)", nh3, list(range(4)), "omol"))
    for mol, size in (("CO", (3, 3, 3)), ("NH3", (3, 3, 3)), ("CO", (4, 4, 4))):
        s = add_molecule_to_slab(build_slab("Pt", "fcc", "111", size), mol, 2.0)
        systems.append((f"{mol}/Pt(111) {size[0]}x{size[1]}x{size[2]} (oc20)", s.atoms, s.adsorbate_indices, "oc20"))

    if args.quick:
        systems, bss = systems[:1], [1, 6]
    rows = []
    for name, atoms, idx, task in systems:
        rows += time_system(name, atoms, idx, task, backbone, head, args.device, bss,
                            max_atoms=args.max_atoms_per_batch)

    out_dir = Path(args.out or f"runs/{datetime.now():%Y%m%d-%H%M%S}_uma_proxy_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "params_total_M": n_params / 1e6, "hyperparameters": UMA_S,
        "total_wall_s": time.perf_counter() - t0,
        "note": "random weights: timing/VRAM proxy only",
        "gpu_guard_gb": float(__import__("os").environ.get("GPU_GUARD_GB", "0") or 0),
        "torch_memory_fraction": frac,
    }
    (out_dir / "benchmark.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"wrote {out_dir / 'benchmark.json'} ; total {meta['total_wall_s']:.1f} s")


if __name__ == "__main__":
    main()
