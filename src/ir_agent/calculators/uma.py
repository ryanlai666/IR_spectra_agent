"""Meta UMA (fairchem) calculator preset and a batched GPU force evaluator.

UMA checkpoints are gated on Hugging Face (``facebook/UMA``). Without an accepted access
request and a token (``hf auth login`` or ``HF_TOKEN``) every entry point raises
:class:`~ir_agent.calculators.errors.CalculatorUnavailableError` with the steps to fix it;
nothing tries to bypass the gate.

API (fairchem-core 2.x, verified against the docs and the installed 2.21.0 source):

    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")   # hub (gated)
    # or, from a local file (what this repo does by default, no HF access):
    # from fairchem.core.units.mlip_unit import load_predict_unit
    # predictor = load_predict_unit("data/checkpoints/uma-s-1p2p1.pt", device="cuda")
    atoms.info.update({"charge": 0, "spin": 1})          # omol task: charge + multiplicity
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")   # "oc20" for slabs
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ir_agent.calculators.errors import CalculatorUnavailableError
from ir_agent.config import REPO_ROOT, UMAConfig
from ir_agent.runtime import get_logger
from ir_agent.vibrations.evaluators import BatchedForces

log = get_logger("uma")

UMA_REPO = "facebook/UMA"
GATE_HELP = (
    "UMA checkpoints are gated on Hugging Face. To enable the 'uma' preset:\n"
    "  1. Log in at https://huggingface.co and request access on https://huggingface.co/facebook/UMA\n"
    "  2. Create a read token at https://huggingface.co/settings/tokens (enable access to public "
    "gated repos)\n"
    "  3. Run `hf auth login` (or set HF_TOKEN) in this environment\n"
    "  4. Re-run: pytest -m 'gpu and uma' tests/test_uma.py"
)


def fairchem_available() -> bool:
    try:
        import fairchem.core  # noqa: F401
    except Exception:  # pragma: no cover - import errors vary by platform
        return False
    return True


def hf_token_available() -> bool:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False
    return bool(get_token())


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except ImportError:
            pass
        log.warning("CUDA requested for UMA but not available; using CPU (slow).")
        return "cpu"
    return requested


#: Presets that pin one UMA task for every system, so an adsorbate shift and its gas-phase
#: reference come from the same task (same DFT level). Plain ``uma`` switches task by system
#: (omol for molecules/clusters, oc20 for slabs): fine for single spectra, wrong for shifts.
UMA_TASK_PRESETS: dict[str, str] = {"uma-omol": "omol", "uma-oc20": "oc20", "uma-omat": "omat"}
UMA_PRESETS: tuple[str, ...] = ("uma", *UMA_TASK_PRESETS)


def task_for_system(system_kind: str, cfg: UMAConfig) -> str:
    """'omol' for isolated molecules/clusters, 'oc20' (configurable) for periodic slabs."""
    return cfg.surface_task if system_kind == "surface" else cfg.molecule_task


def task_for_preset(preset: str, system_kind: str, cfg: UMAConfig) -> str:
    """Task for a UMA preset: fixed for ``uma-<task>``, system-dependent for plain ``uma``."""
    return UMA_TASK_PRESETS.get(preset) or task_for_system(system_kind, cfg)


def local_checkpoint(cfg: UMAConfig) -> Path | None:
    """Absolute path of the configured local checkpoint (None if not configured)."""
    if not cfg.checkpoint_path:
        return None
    p = Path(cfg.checkpoint_path)
    return p if p.is_absolute() else REPO_ROOT / p


def md5sum(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=4)
def _verified_md5(path: str, size: int, mtime: float, expected: str) -> str:
    got = md5sum(Path(path))
    if got.lower() != expected.lower():
        raise CalculatorUnavailableError(
            "uma", f"MD5 mismatch for {path}: got {got}, expected {expected}. Refusing to load."
        )
    return got


def verify_checkpoint(cfg: UMAConfig) -> Path:
    """Return the local checkpoint path after an MD5 check; raise loudly if missing/mismatched."""
    ckpt = local_checkpoint(cfg)
    if ckpt is None or not ckpt.is_file():
        raise CalculatorUnavailableError(
            "uma",
            f"local UMA checkpoint not found at {ckpt}. Place {cfg.model_name}.pt there "
            f"(MD5 {cfg.checkpoint_md5}) or set calculators.uma.allow_hub_download=true. " + GATE_HELP,
        )
    if cfg.checkpoint_md5:
        st = ckpt.stat()
        _verified_md5(str(ckpt), st.st_size, st.st_mtime, cfg.checkpoint_md5)
    return ckpt


def check_uma_ready(cfg: UMAConfig | None = None) -> None:
    """Raise CalculatorUnavailableError unless UMA can be loaded (local checkpoint or HF token)."""
    cfg = cfg or UMAConfig()
    if not fairchem_available():
        raise CalculatorUnavailableError(
            "uma", "fairchem-core is not installed. Run: uv pip install 'fairchem-core==2.21.0'"
        )
    ckpt = local_checkpoint(cfg)
    if ckpt is not None and ckpt.is_file():
        return
    if cfg.allow_hub_download and hf_token_available():
        return
    if cfg.allow_hub_download:
        raise CalculatorUnavailableError("uma", "No Hugging Face token found. " + GATE_HELP)
    verify_checkpoint(cfg)  # raises with the "not found" message


@functools.lru_cache(maxsize=4)
def _load_local_cached(path: str, device: str, inference_settings: str) -> Any:
    """fairchem 2.21 local loading. Skips the gated reference files that get_predict_unit fetches:
    references/iso_atom_elem_refs.yaml (only used for isolated single-atom, non-periodic systems) and
    form_elem_refs.yaml (formation energies). IR spectra use forces only, so nothing is lost."""
    from fairchem.core.units.mlip_unit import load_predict_unit

    return load_predict_unit(path, inference_settings=inference_settings, device=device)


@functools.lru_cache(maxsize=4)
def _load_predictor_cached(model_name: str, device: str, inference_settings: str) -> Any:
    from fairchem.core import pretrained_mlip

    return pretrained_mlip.get_predict_unit(
        model_name, device=device, inference_settings=inference_settings
    )


def load_predictor(cfg: UMAConfig) -> Any:
    """Load (and cache) the UMA predict unit: local MD5-verified checkpoint first, hub only if allowed."""
    check_uma_ready(cfg)
    device = resolve_device(cfg.device)
    from ir_agent.gpu import warn_if_unguarded

    warn_if_unguarded(device)
    ckpt = local_checkpoint(cfg)
    if ckpt is not None and ckpt.is_file():
        path = verify_checkpoint(cfg)
        log.info("Loading UMA from local checkpoint %s (no Hugging Face access)", path)
        return _load_local_cached(str(path), device, cfg.inference_settings)
    try:
        return _load_predictor_cached(cfg.model_name, device, cfg.inference_settings)
    except Exception as exc:  # GatedRepoError, HTTPError 401/403, KeyError for bad names
        msg = str(exc)
        if any(s in msg for s in ("401", "403", "gated", "Gated", "access")):
            raise CalculatorUnavailableError("uma", f"Access to {UMA_REPO} denied. {GATE_HELP}") from exc
        raise CalculatorUnavailableError(
            "uma", f"Could not load UMA model '{cfg.model_name}': {exc}"
        ) from exc


def make_uma_calculator(
    cfg: UMAConfig, system_kind: str, predictor: Any | None = None, task: str | None = None
) -> Any:
    """FAIRChemCalculator with ``task`` (default: the task matching the system, omol or oc20)."""
    from fairchem.core import FAIRChemCalculator

    predictor = predictor if predictor is not None else load_predictor(cfg)
    task = task or task_for_system(system_kind, cfg)
    calc = FAIRChemCalculator(predictor, task_name=task)
    calc.method_label = f"uma ({cfg.model_name}, {task})"
    return calc


def prepare_atoms_for_task(atoms: Atoms, task: str, charge: int = 0, spin: int = 1) -> Atoms:
    """omol needs total charge and spin multiplicity in ``atoms.info``."""
    if task == "omol":
        atoms.info.setdefault("charge", int(charge))
        atoms.info.setdefault("spin", int(spin))
    return atoms


def _default_batch_builder(task: str, predictor: Any) -> Callable[[Sequence[Atoms]], Any]:
    """Build a fairchem AtomicData batch with the same a2g kwargs FAIRChemCalculator uses
    (fairchem-core 2.21 ``ase_calculator.py``: radius 6.0; max_neigh 300 only with external
    graph generation; ``r_data_keys=["spin", "charge"]``)."""
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    settings = getattr(predictor, "inference_settings", None)
    external = bool(getattr(settings, "external_graph_gen", False))
    kwargs = {
        "task_name": task,
        "r_edges": external,
        "r_data_keys": ["spin", "charge"],
        "max_neigh": 300 if external else None,
        "radius": 6.0,
    }
    validate = getattr(predictor, "validate_atoms_data", None)

    def build(images: Sequence[Atoms]) -> Any:
        if validate is not None and images:
            validate(images[0], task)
        data = [AtomicData.from_ase(img, **kwargs) for img in images]
        return atomicdata_list_to_batch(data)

    return build


class UMABatchForces(BatchedForces):
    """Evaluate forces for many displaced geometries in GPU batches.

    All images of a finite-difference run share composition, charge and spin, so they can be
    concatenated into one graph batch; ``batch_size`` / ``max_atoms_per_batch`` bound peak VRAM.

    ``batch_builder`` and ``predictor`` are injectable so the batching logic is unit-testable
    with a mock (no gated weights, no GPU).
    """

    def __init__(
        self,
        predictor: Any,
        task_name: str,
        batch_size: int = 24,
        charge: int = 0,
        spin: int = 1,
        batch_builder: Callable[[Sequence[Atoms]], Any] | None = None,
        max_atoms_per_batch: int | None = None,
    ):
        super().__init__(batch_size, max_atoms_per_batch)
        self.predictor = predictor
        self.task_name = task_name
        self.charge = charge
        self.spin = spin
        self._builder = batch_builder
        self.name = f"uma[{task_name}] (batched x{self.batch_size})"

    def _build(self, images: Sequence[Atoms]) -> Any:
        if self._builder is None:
            self._builder = _default_batch_builder(self.task_name, self.predictor)
        return self._builder(images)

    def prepare(self, img: Atoms) -> Atoms:
        return prepare_atoms_for_task(img, self.task_name, self.charge, self.spin)

    def predict_chunk(self, images: Sequence[Atoms]) -> np.ndarray:
        pred = self.predictor.predict(self._build(images))
        forces = pred["forces"]
        return forces.detach().cpu().numpy() if hasattr(forces, "detach") else np.asarray(forces)
