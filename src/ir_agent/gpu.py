"""Optional GPU guard for shared machines, and the PyTorch memory cap it implies.

If the environment variable ``GPU_GUARD`` points at a guard script, the run scripts in ``scripts/``
launch their CUDA jobs through it::

    python $GPU_GUARD run --agent ir-agent --gb <peak GB> --note "<what>" -- <command>

The guard is expected to export ``GPU_GUARD_GB``; :func:`apply_gpu_guard_cap` turns that into a hard
PyTorch memory cap. When ``GPU_GUARD`` is unset (the default), commands run directly on the GPU.
This module is stdlib-only so the launcher scripts can import it without the scientific stack.
"""

from __future__ import annotations

import os
import sys

from ir_agent.runtime import get_logger

log = get_logger("gpu")

GUARD_ENV = "GPU_GUARD"
GUARD_LABEL = "ir-agent"


def guard_script() -> str | None:
    """Path of the optional GPU guard script (env ``GPU_GUARD``), or None when unset."""
    path = os.environ.get(GUARD_ENV, "").strip()
    return path or None


def guarded(cmd: list[str], gb: float | str, note: str) -> list[str]:
    """Wrap ``cmd`` in the GPU guard if ``GPU_GUARD`` is set; otherwise return it unchanged."""
    script = guard_script()
    if script is None:
        return list(cmd)
    return [sys.executable, script, "run", "--agent", GUARD_LABEL, "--gb", str(gb), "--note", note, "--", *cmd]


def guard_reservation_gb() -> float:
    try:
        return float(os.environ.get("GPU_GUARD_GB", "0") or 0)
    except ValueError:
        return 0.0


def apply_gpu_guard_cap() -> float | None:
    """Cap this process's CUDA caching allocator to the guard reservation. Returns the fraction."""
    gb = guard_reservation_gb()
    if not gb:
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, gb * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    log.info("GPU guard reservation %.2f GB -> torch memory fraction %.3f", gb, frac)
    return frac


def warn_if_unguarded(device: str) -> None:
    """Warn when a guard is configured (``GPU_GUARD``) but this CUDA process was not launched by it."""
    if device.startswith("cuda") and guard_script() and not guard_reservation_gb():
        log.warning("GPU_GUARD is set but this process was not launched through it. Use:\n  %s",
                    " ".join(guarded(["<command>"], "<peak GB>", "<what>")))


def peak_vram_mb() -> float | None:
    """Peak allocated VRAM of this process, without creating a CUDA context as a side effect."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return torch.cuda.max_memory_allocated() / 2**20
