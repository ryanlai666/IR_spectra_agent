"""Orbital Materials Orb models (open weights, Apache-2.0) as calculator presets.

Verified against orb-models 0.7.0 (README, MODELS.md and source, 2026-11-13):

* ``orb-mol``  -> ``orbmol-v2``: OMol25 + OPoly26 (wB97M-V/def2-TZVPD), learnable electrostatics,
  ``ConservativeForcefieldRegressor``; needs ``atoms.info["charge"]`` and ``atoms.info["spin"]``
  (spin multiplicity). Used for gas molecules and the Pt10/Au10 cluster models.
* ``orb-omat`` -> ``orb-v3-conservative-inf-omat``: OMat24 (bulk crystals). The "inf"-neighbour
  conservative variant is the one MODELS.md recommends for Hessian-type quantities (20-neighbour
  models make the PES discontinuous). Slabs are partly out of distribution, molecules fully.

Both are conservative (forces are energy gradients), which finite-difference Hessians need, and are
run with ``precision="float32-highest"`` (orb README: recommended for second/third-order
properties; float64 gave no benefit). ``compile=False`` because torch.compile needs a C++ compiler
(MSVC) on Windows. Graph edges: ``knn_scipy`` on CPU, ``knn_brute_force`` (pure torch) on CUDA;
orb's default ``knn_alchemi`` path uses NVIDIA warp 1.19 (needs a CUDA 13 driver; this machine has
12.8) and a torch.compile'd batching helper (needs MSVC).

orb-models 0.7.0 requires Python >= 3.12 and pins nvalchemi-toolkit-ops < 0.4, so it lives in its
own venv (``.venv-orb``), separate from the fairchem (Python 3.11) and upet environments.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ir_agent.calculators.errors import CalculatorUnavailableError
from ir_agent.config import REPO_ROOT, OrbConfig
from ir_agent.runtime import get_logger
from ir_agent.vibrations.evaluators import BatchedForces

log = get_logger("orb")

#: preset -> (config attribute holding the ORB_PRETRAINED_MODELS key, needs charge/spin)
ORB_PRESETS = {"orb-mol": ("molecule_model", True), "orb-omat": ("periodic_model", False)}

#: known checkpoint file names (orb-models 0.7.0 defaults) for local weights_dir lookup
ORB_CHECKPOINTS = {
    "orbmol-v2": "orbmol-v2-teqabfhg-20260523.ckpt",
    "orb-v3-conservative-inf-omat": "orb-v3-conservative-inf-omat-20250404.ckpt",
    "orb-v3-conservative-omol": "orb-v3-conservative-omol-20250820.ckpt",
}

INSTALL_HINT = (
    "orb-models needs Python >= 3.12. Create the separate env: uv venv .venv-orb --python 3.12; "
    "uv pip install --python .venv-orb torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128; "
    "uv pip install --python .venv-orb orb-models==0.7.0 -e '.[dev]'"
)


def orb_available() -> bool:
    try:
        return importlib.util.find_spec("orb_models") is not None
    except (ImportError, ValueError):
        return False


def model_key(preset: str, cfg: OrbConfig) -> str:
    attr, _ = ORB_PRESETS[preset]
    return getattr(cfg, attr)


def needs_charge_spin(preset: str) -> bool:
    return ORB_PRESETS[preset][1]


def _weights_path(key: str, cfg: OrbConfig) -> str | None:
    if not cfg.weights_dir or key not in ORB_CHECKPOINTS:
        return None
    p = Path(cfg.weights_dir)
    p = p if p.is_absolute() else REPO_ROOT / p
    f = p / ORB_CHECKPOINTS[key]
    return str(f) if f.is_file() else None


@functools.lru_cache(maxsize=4)
def _load_cached(key: str, device: str, precision: str, weights: str | None) -> tuple[Any, Any]:
    from orb_models.forcefield import pretrained

    loader = pretrained.ORB_PRETRAINED_MODELS[key]
    kwargs: dict[str, Any] = {"device": device, "precision": precision, "compile": False}
    if weights:
        kwargs["weights_path"] = weights
    return loader(**kwargs)


def load_model(preset: str, cfg: OrbConfig) -> tuple[Any, Any]:
    """(model, atoms_adapter) for ``preset``; refuses non-conservative models."""
    if not orb_available():
        raise CalculatorUnavailableError(preset, "orb-models not importable here. " + INSTALL_HINT)
    from ir_agent.calculators.uma import resolve_device
    from ir_agent.gpu import warn_if_unguarded

    key = model_key(preset, cfg)
    device = resolve_device(cfg.device)
    warn_if_unguarded(device)
    try:
        model, adapter = _load_cached(key, device, cfg.precision, _weights_path(key, cfg))
    except KeyError as exc:
        raise CalculatorUnavailableError(preset, f"unknown Orb model '{key}'") from exc
    if type(model).__name__ != "ConservativeForcefieldRegressor":
        raise CalculatorUnavailableError(
            preset, f"'{key}' is not a conservative model; finite-difference Hessians need energy gradients"
        )
    return model, adapter


def prepare_atoms(atoms: Atoms, preset: str, charge: int = 0, spin: int = 1) -> Atoms:
    """orbmol needs total charge + multiplicity; the OMat model must not receive them."""
    if needs_charge_spin(preset):
        atoms.info["charge"] = int(atoms.info.get("charge", charge))
        atoms.info["spin"] = int(atoms.info.get("spin", spin))
    else:
        atoms.info.pop("charge", None)
        atoms.info.pop("spin", None)
    return atoms


def edge_method(cfg: OrbConfig, device: Any) -> str:
    """'auto': knn_scipy on CPU (orb refuses it on CUDA), knn_brute_force (pure torch) on CUDA.
    orb's default knn_alchemi needs warp (CUDA 13 driver) + torch.compile (MSVC) on Windows."""
    if cfg.edge_method != "auto":
        return cfg.edge_method
    return "knn_scipy" if str(device).startswith("cpu") else "knn_brute_force"


def make_orb_calculator(preset: str, cfg: OrbConfig) -> Any:
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = load_model(preset, cfg)
    device = next(model.parameters()).device
    calc = ORBCalculator(model, atoms_adapter=adapter, device=device, edge_method=edge_method(cfg, device))
    calc.method_label = preset
    return calc


class OrbBatchForces(BatchedForces):
    """Batched conservative forces: per-image graphs (CPU knn) -> ``adapter.batch`` -> one predict.

    ``graph_builder`` / ``predict`` are injectable for unit tests without orb-models.
    """

    def __init__(
        self,
        preset: str,
        model: Any,
        adapter: Any,
        cfg: OrbConfig,
        charge: int = 0,
        spin: int = 1,
        predict: Callable[[Sequence[Atoms]], np.ndarray] | None = None,
    ):
        super().__init__(cfg.batch_size, cfg.max_atoms_per_batch)
        self.preset, self.model, self.adapter, self.cfg = preset, model, adapter, cfg
        self.charge, self.spin = charge, spin
        self._predict = predict
        self.name = f"{preset}[{model_key(preset, cfg)}] (batched x{self.batch_size})"

    def prepare(self, img: Atoms) -> Atoms:
        return prepare_atoms(img, self.preset, self.charge, self.spin)

    def predict_chunk(self, images: Sequence[Atoms]) -> np.ndarray:
        if self._predict is not None:
            return self._predict(images)
        device = next(self.model.parameters()).device
        em = edge_method(self.cfg, device)
        graphs = [self.adapter.from_ase_atoms(img, edge_method=em, device=device) for img in images]
        batch = self.adapter.batch(graphs).to(device)
        out = self.model.predict(batch)
        return out[self.model.grad_forces_name].detach().cpu().numpy()


def make_orb_force_evaluator(preset: str, cfg: OrbConfig, charge: int = 0, spin: int = 1) -> OrbBatchForces:
    model, adapter = load_model(preset, cfg)
    return OrbBatchForces(preset, model, adapter, cfg, charge, spin)
