"""Typed configuration for ir-agent.

Defaults live in the pydantic models below; ``configs/default.yaml`` (or any YAML passed
with ``--config``) overrides them, and a few environment variables override both.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Section):
    output_root: str = "runs"
    log_level: str = "INFO"


class LLMConfig(_Section):
    provider: Literal["google"] = "google"
    model: str = "gemini-3.8-flash"
    fallback_models: list[str] = Field(default_factory=lambda: ["gemini-3.6-flash"])
    temperature: float = 0.0
    max_retries: int = 3
    timeout_s: float = 60.0


class UMAConfig(_Section):
    #: label (+ fairchem registry name for hub loading); uma-s-1p2p1 = UMA-S 1.2.1. fairchem 2.21
    #: lists uma-s-1p2 but not 1p2p1, so this checkpoint is only used through checkpoint_path
    #: (load_predict_unit verified on CPU).
    model_name: str = "uma-s-1p2p1"
    #: local checkpoint (downloaded manually from the gated repo, MD5-checked); loaded with
    #: fairchem's load_predict_unit -> no Hugging Face access at all
    checkpoint_path: str | None = "data/checkpoints/uma-s-1p2p1.pt"
    checkpoint_md5: str | None = "3497615fd30a24c5b35cd3b41a682e6e"
    #: fall back to get_predict_unit(model_name) (gated download, needs `hf auth login`)
    allow_hub_download: bool = False
    device: str = "cuda"
    molecule_task: str = "omol"
    surface_task: str = "oc20"
    inference_settings: str = "default"
    batch_size: int = 24
    #: cap on atoms per GPU forward pass (memory scales with edges; ~180 atoms of a dense Pt slab
    #: peaked at ~2.46 GiB with a UMA-S-sized model on the RTX 3070, see runs/uma_proxy_benchmark)
    max_atoms_per_batch: int = 180


class OrbConfig(_Section):
    """Orbital Materials Orb models (orb-models >= 0.7, Python >= 3.12: use .venv-orb)."""

    #: ORB_PRETRAINED_MODELS keys. Both are *conservative* (forces = -dE/dx), as Hessians need.
    molecule_model: str = "orbmol-v2"  # OMol25 + OPoly26, needs charge + spin multiplicity
    periodic_model: str = "orb-v3-conservative-inf-omat"  # OMat24 bulk; slabs/molecules are OOD
    weights_dir: str | None = "data/raw/orb"  # local .ckpt files; None = download via cached_path
    device: str = "cpu"
    precision: Literal["float32-high", "float32-highest", "float64"] = "float32-highest"
    #: "auto" = knn_scipy on CPU, knn_brute_force (pure torch) on CUDA; avoids orb's default
    #: knn_alchemi (warp 1.19 needs a CUDA 13 driver; its batch helper needs MSVC for torch.compile)
    edge_method: str = "auto"
    batch_size: int = 16
    max_atoms_per_batch: int = 180


class PETConfig(_Section):
    """Lab COSMO UPET/PET models (upet >= 0.3, Python >= 3.12: use .venv-pet)."""

    molecule_checkpoint: str = "pet-omol-l-v1.0.0.ckpt"  # OMol25 (wB97M-V); reads charge/spin
    periodic_checkpoint: str = "pet-mad-s-v1.6.0.ckpt"  # MAD-1.6 (r2SCAN, incl. catalytic surfaces)
    models_dir: str = "data/raw/pet"
    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float32"
    batch_size: int = 16
    max_atoms_per_batch: int = 180


class XTBConfig(_Section):
    binary: str | None = None
    method: Literal["gfn0", "gfn1", "gfn2"] = "gfn2"
    accuracy: float = 1.0
    electronic_temperature: float = 300.0
    timeout_s: float = 600.0


class DXTBConfig(_Section):
    method: Literal["gfn1", "gfn2"] = "gfn1"
    device: str = "cpu"


class DFTBConfig(_Section):
    skf_dir: str | None = None


class GPAWConfig(_Section):
    pw_cutoff_ev: float = 200.0
    xc: str = "LDA"


class CalculatorsConfig(_Section):
    default: str = "emt"
    uma: UMAConfig = Field(default_factory=UMAConfig)
    orb: OrbConfig = Field(default_factory=OrbConfig)
    pet: PETConfig = Field(default_factory=PETConfig)
    xtb: XTBConfig = Field(default_factory=XTBConfig)
    dxtb: DXTBConfig = Field(default_factory=DXTBConfig)
    dftb: DFTBConfig = Field(default_factory=DFTBConfig)
    gpaw: GPAWConfig = Field(default_factory=GPAWConfig)


class GeometryConfig(_Section):
    slab_size: tuple[int, int, int] = (3, 3, 3)
    vacuum: float = 10.0
    adsorbate_height: float = 2.0
    site: str = "ontop"
    fix_bottom_layers: int = 2
    molecule_vacuum: float = 8.0
    cluster_radius: float = 3.0  #: lateral cut-off (A) for cluster models of a surface site
    #: if set, the cut-off is this multiple of the slab's nearest-neighbour distance instead
    #: (1.083 reproduces Pt10/Au10 at 3.0 A and gives the same 7 + 3 cluster for Cu and Ru)
    cluster_radius_nn: float | None = None
    cluster_layers: int = 2


class OptimizationConfig(_Section):
    optimizer: Literal["BFGSLineSearch", "BFGS", "LBFGS", "FIRE"] = "BFGS"
    fmax_preopt: float = 0.05
    fmax_final: float = 0.01
    max_steps: int = 400
    stall_steps: int = 30  #: stop if fmax has not improved by 1 % for this many steps


class VibrationsConfig(_Section):
    delta: float = 0.01
    nfree: Literal[2, 4] = 2
    indices: Literal["adsorbate", "all"] = "adsorbate"
    project_rigid_modes: bool = True
    min_frequency_cm: float = 50.0


class CriticConfig(_Section):
    """Critic node after the calculation (see ir_agent.critic)."""

    enabled: bool = True
    max_retries: int = Field(default=1, ge=0)
    imag_threshold_cm: float = 50.0  #: modes below -this count as imaginary
    follow_amplitude: float = 0.05  #: largest atomic step (A) along an imaginary mode
    retry_optimizer: Literal["BFGSLineSearch", "BFGS", "LBFGS", "FIRE"] = "FIRE"


class IRConfig(_Section):
    dipole_method: Literal["auto", "native", "xtb", "xtb-gfn1", "tblite", "dxtb", "none"] = "auto"
    surface_selection_rule: bool = True
    image_dipole_factor: float = 1.0


class SpectrumConfig(_Section):
    lineshape: Literal["lorentzian", "gaussian"] = "lorentzian"
    fwhm_cm: float = 20.0
    x_min_cm: float = 400.0
    x_max_cm: float = 4000.0
    n_points: int = 3601
    scale_factor: float = 1.0


class TrackingConfig(_Section):
    mlflow: bool = True
    tracking_uri: str = "mlruns"
    experiment: str = "ir-agent"


class AppConfig(_Section):
    run: RunConfig = Field(default_factory=RunConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    calculators: CalculatorsConfig = Field(default_factory=CalculatorsConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    vibrations: VibrationsConfig = Field(default_factory=VibrationsConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    ir: IRConfig = Field(default_factory=IRConfig)
    spectrum: SpectrumConfig = Field(default_factory=SpectrumConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "IR_AGENT_LLM_MODEL": ("llm", "model"),
    "IR_AGENT_XTB_BIN": ("calculators", "xtb", "binary"),
    "IR_AGENT_UMA_MODEL": ("calculators", "uma", "model_name"),
    "IR_AGENT_DEVICE": ("calculators", "uma", "device"),
    "IR_AGENT_ORB_DEVICE": ("calculators", "orb", "device"),
    "IR_AGENT_PET_DEVICE": ("calculators", "pet", "device"),
    "IR_AGENT_OUTPUT_ROOT": ("run", "output_root"),
}


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> AppConfig:
    """Load the configuration.

    Precedence (highest first): ``overrides`` > environment variables > YAML file > defaults.
    If ``path`` is None the repo's ``configs/default.yaml`` is used when it exists.
    """
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if use_env:
        for env, keys in _ENV_OVERRIDES.items():
            val = os.environ.get(env)
            if val:
                node = data
                for k in keys[:-1]:
                    node = node.setdefault(k, {})
                node[keys[-1]] = val

    if overrides:
        data = _deep_update(data, overrides)
    return AppConfig.model_validate(data)
