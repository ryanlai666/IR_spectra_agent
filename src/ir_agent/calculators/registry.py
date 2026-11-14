"""Calculator presets: availability checks, factories and evaluator construction.

Unlike v0.1, a missing calculator is never silently replaced by EMT: callers get a
:class:`CalculatorUnavailableError` that says what to install.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator

from ir_agent.calculators.errors import CalculatorUnavailableError, UnsupportedSystemError
from ir_agent.calculators.uma import UMA_PRESETS
from ir_agent.config import AppConfig
from ir_agent.runtime import get_logger
from ir_agent.vibrations.evaluators import ASECalculatorDipoles, ASECalculatorForces

log = get_logger("calculators")

SystemKind = Literal["molecule", "cluster", "surface"]

EMT_ELEMENTS = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"}


@dataclass(frozen=True)
class PresetInfo:
    name: str
    method: str
    periodic: bool  #: can treat periodic slabs
    native_dipole: bool  #: calculator returns dipole moments itself
    gpu: bool
    platforms: str
    install: str
    note: str = ""


PRESETS: dict[str, PresetInfo] = {
    "emt": PresetInfo(
        "emt", "Effective medium theory (ASE built-in)", True, False, False, "all",
        "built into ASE", "Toy potential; only Al Cu Ag Au Ni Pd Pt H C N O. Workflow testing only.",
    ),
    "uma": PresetInfo(
        "uma", "Meta UMA MLIP via fairchem (omol / oc20 tasks)", True, False, True, "all",
        "uv pip install fairchem-core==2.21.0 ; request access to facebook/UMA ; hf auth login",
        "Gated checkpoints. No dipoles: IR intensities use the hybrid xTB-dipole scheme. Task "
        "switches with the system; use a uma-<task> preset for adsorption shifts.",
    ),
    "uma-omol": PresetInfo(
        "uma-omol", "UMA, omol task only (molecules + Pt10/Au10 cluster models)", False, False, True,
        "all", "as for uma",
        "Same task for gas reference and adsorbate: shifts at one DFT level (wB97M-V).",
    ),
    "uma-oc20": PresetInfo(
        "uma-oc20", "UMA, oc20 task only (molecules + periodic slabs)", True, False, True,
        "all", "as for uma",
        "Gas reference also with oc20 (RPBE level), so the shift is within one task.",
    ),
    "uma-omat": PresetInfo(
        "uma-omat", "UMA, omat task only (molecules + periodic slabs)", True, False, True,
        "all", "as for uma",
        "Bulk-materials task (PBE/PBE+U); molecules are out of its training distribution.",
    ),
    "orb-mol": PresetInfo(
        "orb-mol", "Orb OrbMol-v2 MLIP (OMol25+OPoly26, conservative)", False, False, True,
        "all (Python >= 3.12: .venv-orb)", "uv pip install orb-models==0.7.0 (see README, .venv-orb)",
        "Molecules and Pt10/Au10 cluster models; needs charge + spin. Open weights (Apache-2.0).",
    ),
    "orb-omat": PresetInfo(
        "orb-omat", "Orb v3 conservative-inf OMat24 MLIP", True, False, True,
        "all (Python >= 3.12: .venv-orb)", "uv pip install orb-models==0.7.0 (see README, .venv-orb)",
        "Periodic slabs (bulk-trained: slabs partly, molecules fully out of distribution).",
    ),
    "pet-omol": PresetInfo(
        "pet-omol", "PET-OMol (UPET, OMol25 wB97M-V)", False, False, True,
        "all (Python >= 3.12: .venv-pet)", "uv pip install upet==0.3.1 (see README, .venv-pet) + checkpoint",
        "Molecules and cluster models; reads charge + spin. Open weights (BSD-3-Clause).",
    ),
    "pet-mad": PresetInfo(
        "pet-mad", "PET-MAD-1.6 (UPET, r2SCAN, incl. catalytic surfaces)", True, False, True,
        "all (Python >= 3.12: .venv-pet)", "uv pip install upet==0.3.1 (see README, .venv-pet) + checkpoint",
        "Periodic slabs.",
    ),
    "xtb": PresetInfo(
        "xtb", "GFN2-xTB via the xtb executable", False, True, False, "all (conda-forge win-64/linux-64)",
        "conda install -c conda-forge xtb  (or set IR_AGENT_XTB_BIN)",
        "Non-periodic only: slabs are converted to a cluster model.",
    ),
    "xtb-gfn1": PresetInfo(
        "xtb-gfn1", "GFN1-xTB via the xtb executable", False, True, False, "all (conda-forge win-64/linux-64)",
        "conda install -c conda-forge xtb  (or set IR_AGENT_XTB_BIN)",
        "Converges for metallic Au clusters where GFN2 SCC fails; non-periodic only.",
    ),
    "tblite": PresetInfo(
        "tblite", "GFN2-xTB via tblite python API", True, True, False,
        "linux/macOS (pip) ; Windows via conda-forge tblite-python",
        "pip install tblite  |  conda install -c conda-forge tblite-python",
    ),
    "dxtb": PresetInfo(
        "dxtb", "GFN1-xTB in PyTorch (dxtb)", False, True, True, "all (pip)",
        "uv pip install dxtb", "Molecules/clusters only; GFN2 needs tad-libcint (Linux).",
    ),
    "dftb+": PresetInfo(
        "dftb+", "DFTB+ (SCC-DFTB, auorg/3ob parameters)", True, False, False, "linux/WSL only",
        "conda install -c conda-forge 'dftbplus=*=nompi_*' ; export DFTB_PREFIX=<skf dir>",
        "No win-64 conda build (checked 2026-11-12).",
    ),
    "gpaw": PresetInfo(
        "gpaw", "GPAW plane-wave DFT", True, True, False, "linux/WSL only",
        "conda install -c conda-forge gpaw",
    ),
    "nwchem": PresetInfo(
        "nwchem", "NWChem plane-wave DFT", True, False, False, "linux/WSL only",
        "conda install -c conda-forge nwchem",
    ),
    "vasp": PresetInfo(
        "vasp", "VASP PBE (medium)", True, True, False, "HPC (licensed)",
        "set ASE_VASP_COMMAND (or VASP_COMMAND) and VASP_PP_PATH", "Untested in this repo.",
    ),
    "vasp-accurate": PresetInfo(
        "vasp-accurate", "VASP PBE-D3 (tight)", True, True, False, "HPC (licensed)",
        "set ASE_VASP_COMMAND (or VASP_COMMAND) and VASP_PP_PATH", "Untested in this repo.",
    ),
}

ALIASES = {
    "fast": "emt",
    "cheap": "emt",
    "gfn2": "xtb",
    "gfn2-xtb": "xtb",
    "gfn1": "dxtb",
    "gfn1-xtb": "dxtb",
    "fairchem": "uma",
    "orb": "orb-mol",
    "orbmol": "orb-mol",
    "orbmol-v2": "orb-mol",
    "orb-v3": "orb-omat",
    "pet": "pet-omol",
    "upet": "pet-omol",
    "pet-omol-l": "pet-omol",
    "pet-mad-s": "pet-mad",
    "mlip": "uma",
    "dftb": "dftb+",
    "dftbplus": "dftb+",
    "paw": "gpaw",
    "default": "vasp",
    "dft": "vasp",
    "accurate": "vasp-accurate",
    "vasp-high": "vasp-accurate",
}


def canonical_preset(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in PRESETS:
        raise ValueError(f"Unknown calculator preset '{name}'. Known: {sorted(PRESETS)}")
    return key


def _has_module(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def preset_status(name: str, cfg: AppConfig) -> tuple[bool, str]:
    """(available, reason) without loading heavy models."""
    key = canonical_preset(name)
    if key == "emt":
        return True, "built-in"
    if key in UMA_PRESETS:
        from ir_agent.calculators.uma import (
            fairchem_available,
            hf_token_available,
            local_checkpoint,
        )

        u = cfg.calculators.uma
        if not fairchem_available():
            return False, "fairchem-core not installed"
        ckpt = local_checkpoint(u)
        if ckpt is not None and ckpt.is_file():
            return True, f"local checkpoint {ckpt.name} ({u.model_name})"
        if u.allow_hub_download and hf_token_available():
            return True, f"fairchem + HF token ({u.model_name})"
        return False, f"no local checkpoint at {u.checkpoint_path} (gated facebook/UMA)"
    if key in ("orb-mol", "orb-omat"):
        from ir_agent.calculators.orb import model_key, orb_available

        ok = orb_available()
        return ok, (model_key(key, cfg.calculators.orb) if ok else "orb-models not importable (use .venv-orb)")
    if key in ("pet-omol", "pet-mad"):
        from ir_agent.calculators.pet import status

        return status(key, cfg.calculators.pet)
    if key in ("xtb", "xtb-gfn1"):
        from ir_agent.calculators.xtb_binary import find_xtb_binary

        b = find_xtb_binary(cfg.calculators.xtb.binary)
        return (b is not None), (str(b) if b else "xtb executable not found")
    if key == "tblite":
        return _has_module("tblite"), "tblite python module" if _has_module("tblite") else "tblite not importable"
    if key == "dxtb":
        return _has_module("dxtb"), "dxtb" if _has_module("dxtb") else "dxtb not installed"
    if key == "dftb+":
        ok = shutil.which("dftb+") is not None and bool(cfg.calculators.dftb.skf_dir or os.environ.get("DFTB_PREFIX"))
        return ok, "dftb+ + DFTB_PREFIX" if ok else "dftb+ binary or DFTB_PREFIX missing (Linux/WSL only)"
    if key == "gpaw":
        return _has_module("gpaw"), "gpaw" if _has_module("gpaw") else "gpaw not installed (Linux/WSL)"
    if key == "nwchem":
        ok = shutil.which("nwchem") is not None
        return ok, "nwchem" if ok else "nwchem binary not found (Linux/WSL)"
    if key in ("vasp", "vasp-accurate"):
        cmd = os.environ.get("ASE_VASP_COMMAND") or os.environ.get("VASP_COMMAND")
        ok = bool(cmd and os.environ.get("VASP_PP_PATH"))
        return ok, "VASP env set" if ok else "ASE_VASP_COMMAND/VASP_COMMAND or VASP_PP_PATH not set"
    return False, "unknown"  # pragma: no cover


def list_presets(cfg: AppConfig) -> list[dict[str, Any]]:
    rows = []
    for key, info in PRESETS.items():
        ok, reason = preset_status(key, cfg)
        rows.append({**asdict(info), "available": ok, "status": reason})
    return rows


def require(name: str, cfg: AppConfig) -> str:
    key = canonical_preset(name)
    ok, reason = preset_status(key, cfg)
    if not ok:
        raise CalculatorUnavailableError(key, f"{reason}. Install: {PRESETS[key].install}")
    return key


def make_calculator(
    name: str,
    cfg: AppConfig,
    *,
    atoms: Atoms,
    system_kind: SystemKind = "molecule",
    mode: Literal["opt", "freq"] = "opt",
    charge: int = 0,
    spin: int = 1,
    workdir: Path | None = None,
) -> Calculator:
    """Instantiate the ASE calculator for preset ``name`` (raises if unavailable)."""
    key = require(name, cfg)
    info = PRESETS[key]
    periodic = bool(atoms.pbc.any())
    if periodic and not info.periodic:
        raise UnsupportedSystemError(
            f"Preset '{key}' cannot treat periodic systems; use surface.model='cluster' or a "
            "periodic preset (uma, emt, tblite)."
        )
    if key == "emt":
        bad = sorted(set(atoms.get_chemical_symbols()) - EMT_ELEMENTS)
        if bad:
            raise UnsupportedSystemError(f"EMT has no parameters for {bad}; supported: {sorted(EMT_ELEMENTS)}")
        from ase.calculators.emt import EMT

        return EMT()
    if key in UMA_PRESETS:
        from ir_agent.calculators.uma import (
            make_uma_calculator,
            prepare_atoms_for_task,
            task_for_preset,
        )

        kind = "surface" if system_kind == "surface" else "molecule"
        task = task_for_preset(key, kind, cfg.calculators.uma)
        prepare_atoms_for_task(atoms, task, charge, spin)
        return make_uma_calculator(cfg.calculators.uma, kind, task=task)
    if key in ("orb-mol", "orb-omat"):
        from ir_agent.calculators.orb import make_orb_calculator, prepare_atoms

        prepare_atoms(atoms, key, charge, spin)
        return make_orb_calculator(key, cfg.calculators.orb)
    if key in ("pet-omol", "pet-mad"):
        from ir_agent.calculators.pet import make_pet_calculator, prepare_atoms

        prepare_atoms(atoms, key, charge, spin)
        return make_pet_calculator(key, cfg.calculators.pet)
    if key in ("xtb", "xtb-gfn1"):
        from ir_agent.calculators.xtb_binary import XTBBinary

        x = cfg.calculators.xtb
        calc = XTBBinary(
            binary=x.binary, method="gfn1" if key == "xtb-gfn1" else x.method, charge=charge,
            uhf=spin - 1, accuracy=x.accuracy,
            electronic_temperature=x.electronic_temperature, timeout_s=x.timeout_s,
            scratch_dir=(workdir / "xtb_scratch") if workdir else None,
        )
        calc.name = key
        return calc
    if key == "tblite":
        from tblite.ase import TBLite

        return TBLite(method="GFN2-xTB", charge=charge, multiplicity=spin, verbosity=0)
    if key == "dxtb":
        from ir_agent.calculators.dxtb_calc import DXTBCalculator

        d = cfg.calculators.dxtb
        return DXTBCalculator(method=d.method, charge=charge, spin=spin, device=d.device)
    if key == "dftb+":
        return _make_dftb(cfg, mode, workdir)
    if key == "gpaw":
        from gpaw import GPAW, PW

        g = cfg.calculators.gpaw
        kw: dict[str, Any] = {"mode": PW(g.pw_cutoff_ev), "xc": g.xc, "kpts": (1, 1, 1), "txt": None}
        if mode == "freq":
            kw["symmetry"] = "off"
        return GPAW(**kw)
    if key == "nwchem":
        from ase.calculators.nwchem import NWChem

        return NWChem(theory="band", xc="LDA", set={"pspw:wcut": 10.0}, kpts=(1, 1, 1),
                      directory=str(workdir / "nwchem") if workdir else "nwchem")
    if key in ("vasp", "vasp-accurate"):
        return _make_vasp(key, atoms, mode, workdir)
    raise CalculatorUnavailableError(key, "no factory")  # pragma: no cover


def _make_dftb(cfg: AppConfig, mode: str, workdir: Path | None) -> Calculator:
    from ase.calculators.dftb import Dftb

    skf = cfg.calculators.dftb.skf_dir or os.environ.get("DFTB_PREFIX")
    if skf:
        os.environ["DFTB_PREFIX"] = skf
    return Dftb(
        label="dftb",
        directory=str(workdir / "dftb") if workdir else ".",
        kpts=(1, 1, 1),
        Hamiltonian_MaxAngularMomentum_="",
        Hamiltonian_MaxAngularMomentum_O="p",
        Hamiltonian_MaxAngularMomentum_H="s",
        Hamiltonian_MaxAngularMomentum_N="p",
        Hamiltonian_MaxAngularMomentum_C="p",
        Hamiltonian_MaxAngularMomentum_Au="d",
        Hamiltonian_MaxAngularMomentum_S="d",
        Hamiltonian_MaxAngularMomentum_P="d",
        Hamiltonian_MaxSCCIterations=1000 if mode == "freq" else 500,
    )


def _make_vasp(key: str, atoms: Atoms, mode: str, workdir: Path | None) -> Calculator:
    from ase.calculators.vasp import Vasp

    common: dict[str, Any] = {"xc": "PBE", "ismear": 0, "sigma": 0.05,
                              "directory": str(workdir / "vasp") if workdir else "vasp"}
    if key == "vasp-accurate":
        common.update(encut=500, kpts=(4, 4, 1), ivdw=12, lreal=False, prec="Accurate")
    else:
        common.update(encut=400, kpts=(2, 2, 1), prec="Normal")
    if mode == "freq":
        common.update(ediff=1e-8 if key == "vasp-accurate" else 1e-6, isym=0)
        if atoms.pbc.any():  # dipole correction along the surface normal
            common.update(ldipol=True, idipol=3, dipol=atoms.get_center_of_mass(scaled=True).tolist())
    return Vasp(**common)


# ----------------------------------------------------------------------------------------------
# Evaluators for the finite-difference IR driver
# ----------------------------------------------------------------------------------------------


def make_force_evaluator(
    name: str,
    cfg: AppConfig,
    *,
    atoms: Atoms,
    system_kind: SystemKind,
    charge: int = 0,
    spin: int = 1,
    workdir: Path | None = None,
):
    """Batched UMA evaluator on GPU, otherwise a loop over the preset's ASE calculator."""
    key = canonical_preset(name)
    if key in UMA_PRESETS:
        from ir_agent.calculators.uma import UMABatchForces, load_predictor, task_for_preset

        kind = "surface" if system_kind == "surface" else "molecule"
        u = cfg.calculators.uma
        return UMABatchForces(load_predictor(u), task_for_preset(key, kind, u), u.batch_size, charge, spin,
                              max_atoms_per_batch=u.max_atoms_per_batch)
    if key in ("orb-mol", "orb-omat"):
        from ir_agent.calculators.orb import make_orb_force_evaluator, prepare_atoms

        prepare_atoms(atoms, key, charge, spin)
        return make_orb_force_evaluator(key, cfg.calculators.orb, charge, spin)
    calc = make_calculator(key, cfg, atoms=atoms, system_kind=system_kind, mode="freq",
                           charge=charge, spin=spin, workdir=workdir)
    return ASECalculatorForces(calc, name=key)


DIPOLE_AUTO_ORDER = ("xtb", "tblite", "dxtb")


def resolve_dipole_method(method: str, force_preset: str, cfg: AppConfig) -> str | None:
    """Pick the dipole backend. 'auto' prefers the force preset itself if it has native dipoles,
    then GFN2-xTB (xtb binary, tblite), then GFN1-xTB (dxtb). Returns None if nothing usable."""
    force_key = canonical_preset(force_preset)
    if method == "none":
        return None
    if method == "native":
        if not PRESETS[force_key].native_dipole:
            raise UnsupportedSystemError(f"Preset '{force_key}' has no native dipoles")
        return force_key
    if method != "auto":
        require(method, cfg)
        return method
    if PRESETS[force_key].native_dipole and preset_status(force_key, cfg)[0]:
        return force_key
    for cand in DIPOLE_AUTO_ORDER:
        if preset_status(cand, cfg)[0]:
            return cand
    return None


def make_dipole_evaluator(
    method: str,
    cfg: AppConfig,
    *,
    charge: int = 0,
    spin: int = 1,
    region: Sequence[int] | None = None,
    cluster_builder: Callable[[Atoms], Atoms] | None = None,
    workdir: Path | None = None,
):
    """Dipole evaluator for ``method`` in {xtb, tblite, dxtb}, optionally on a fragment."""
    probe = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], pbc=False)
    calc = make_calculator(method, cfg, atoms=probe, system_kind="molecule", mode="freq",
                           charge=charge, spin=spin, workdir=workdir)
    label = {"xtb": "GFN2-xTB (xtb)", "xtb-gfn1": "GFN1-xTB (xtb)", "tblite": "GFN2-xTB (tblite)",
             "dxtb": "GFN1-xTB (dxtb)"}.get(method, method)
    if region is not None or cluster_builder is not None:
        label += " on adsorbate fragment" if cluster_builder is None else " on cluster"
    return ASECalculatorDipoles(calc, name=label, region=region, cluster_builder=cluster_builder)


def platform_summary() -> str:
    return f"{sys.platform} / python {sys.version.split()[0]}"
