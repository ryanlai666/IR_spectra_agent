"""Lab COSMO UPET / PET models (open weights, BSD-3-Clause, huggingface.co/lab-cosmo/upet).

Verified against upet 0.3.1 / metatomic-ase (2026-11-13):

* ``pet-omol`` -> ``pet-omol-l-v1.0.0.ckpt`` (OMol25, wB97M-V; S/M/L sizes exist). Loaded with
  ``upet.ase.UPETCalculator(checkpoint_path=..., device=...)``. The model requests total charge and
  spin multiplicity, which ``metatomic_ase`` reads from ``atoms.info["charge"]`` (default 0) and
  ``atoms.info["spin_multiplicity"]`` or ``atoms.info["spin"]`` (default 1) - checked in the source
  and by an energy change for charge=1/spin=2. Used for gas molecules and Pt10/Au10 cluster models.
* ``pet-mad`` -> ``pet-mad-s-v1.6.0.ckpt`` (PET-MAD-1.6, r2SCAN, 102 elements, additionally trained
  on catalytic surfaces per the upet README). Used for periodic metal slabs.

Forces are conservative by default (``non_conservative=False`` = energy gradients), which the
finite-difference Hessian needs. upet 0.3.1 pins nvalchemi-toolkit-ops >= 0.4 (orb-models needs
< 0.4) and Python >= 3.11; it lives in ``.venv-pet`` (Python 3.12, torch 2.8 cu128).

Batching: metatomic's batched ``compute_energy`` does not attach the model's requested inputs
(charge/spin) to each system, only the single-structure ``calculate`` does, so displaced
geometries are evaluated one by one through the ASE calculator.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
from typing import Any

from ase import Atoms

from ir_agent.calculators.errors import CalculatorUnavailableError
from ir_agent.config import REPO_ROOT, PETConfig
from ir_agent.runtime import get_logger

log = get_logger("pet")

#: preset -> (config attribute holding the checkpoint file name, model reads charge/spin)
PET_PRESETS = {"pet-omol": ("molecule_checkpoint", True), "pet-mad": ("periodic_checkpoint", False)}

INSTALL_HINT = (
    "upet needs its own env: uv venv .venv-pet --python 3.12; uv pip install --python .venv-pet "
    "torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128; uv pip install --python "
    ".venv-pet upet==0.3.1 -e '.[dev]'; checkpoints from https://huggingface.co/lab-cosmo/upet "
    "(models/*.ckpt) into data/raw/pet"
)


def pet_available() -> bool:
    try:
        return importlib.util.find_spec("upet") is not None
    except (ImportError, ValueError):
        return False


def checkpoint_path(preset: str, cfg: PETConfig) -> Path:
    attr, _ = PET_PRESETS[preset]
    d = Path(cfg.models_dir)
    d = d if d.is_absolute() else REPO_ROOT / d
    return d / getattr(cfg, attr)


def reads_charge_spin(preset: str) -> bool:
    return PET_PRESETS[preset][1]


def status(preset: str, cfg: PETConfig) -> tuple[bool, str]:
    if not pet_available():
        return False, "upet not importable (use .venv-pet)"
    ckpt = checkpoint_path(preset, cfg)
    if not ckpt.is_file():
        return False, f"checkpoint missing: {ckpt}"
    return True, ckpt.name


def _disable_nvalchemi_without_warp_cuda() -> bool:
    """metatomic_ase uses nvalchemiops (NVIDIA warp) for neighbour lists of CUDA systems.
    nvalchemi-toolkit-ops 0.4 needs warp >= 1.13, whose CUDA backend needs a CUDA 13 driver; with
    an older driver warp sees no CUDA device and the call fails (IndexError). In that case switch
    metatomic to its vesin neighbour lists. Returns True if the switch was made."""
    try:
        import metatomic_ase._neighbors as nb
    except ImportError:
        return False
    if not getattr(nb, "HAS_NVALCHEMIOPS", False):
        return False
    try:
        import warp as wp

        wp.init()
        if wp.get_cuda_device_count() > 0:
            return False
    except Exception:  # pragma: no cover - warp missing or broken
        pass
    nb.HAS_NVALCHEMIOPS = False
    log.warning("warp has no usable CUDA device (needs a CUDA 13 driver): metatomic falls back to "
                "vesin neighbour lists")
    return True


@functools.lru_cache(maxsize=4)
def _load_cached(path: str, device: str, dtype: str) -> Any:
    import torch
    from upet.ase import UPETCalculator

    return UPETCalculator(
        checkpoint_path=path,
        device=device,
        dtype=torch.float64 if dtype == "float64" else torch.float32,
        non_conservative=False,
    )


def make_pet_calculator(preset: str, cfg: PETConfig) -> Any:
    ok, reason = status(preset, cfg)
    if not ok:
        raise CalculatorUnavailableError(preset, f"{reason}. {INSTALL_HINT}")
    from ir_agent.calculators.uma import resolve_device
    from ir_agent.gpu import warn_if_unguarded

    device = resolve_device(cfg.device)
    warn_if_unguarded(device)
    if device.startswith("cuda"):
        _disable_nvalchemi_without_warp_cuda()
    calc = _load_cached(str(checkpoint_path(preset, cfg)), device, cfg.dtype)
    calc.method_label = preset
    return calc


def prepare_atoms(atoms: Atoms, preset: str, charge: int = 0, spin: int = 1) -> Atoms:
    """pet-omol reads total charge + multiplicity from atoms.info; set them explicitly."""
    if reads_charge_spin(preset):
        atoms.info["charge"] = int(atoms.info.get("charge", charge))
        atoms.info["spin"] = int(atoms.info.get("spin", spin))
    return atoms
