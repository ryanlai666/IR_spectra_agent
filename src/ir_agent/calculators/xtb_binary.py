"""Minimal ASE calculator around the ``xtb`` executable (GFN0/1/2-xTB).

Works on Windows (conda-forge ``xtb`` win-64 build) and Linux without the xtb-python bindings.
Provides energy, forces, dipole and partial charges for non-periodic systems, plus xtb's own
analytical-Hessian IR spectrum (``--hess``) used to cross-check the hybrid finite-difference
implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorError, all_changes
from ase.io import write
from ase.units import Bohr, Hartree

from ir_agent.config import REPO_ROOT
from ir_agent.runtime import get_logger

log = get_logger("xtb")


def find_xtb_binary(explicit: str | None = None) -> Path | None:
    """Locate the xtb executable: explicit path > $IR_AGENT_XTB_BIN > PATH > ./.tools/xtb."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("IR_AGENT_XTB_BIN")
    if env:
        candidates.append(Path(env))
    which = shutil.which("xtb")
    if which:
        candidates.append(Path(which))
    tools = REPO_ROOT / ".tools" / "xtb"
    if sys.platform.startswith("win"):
        candidates += [tools / "Library" / "bin" / "xtb.exe", tools / "Scripts" / "xtb.exe"]
    else:
        candidates.append(tools / "bin" / "xtb")
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _subprocess_env(binary: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_STACKSIZE", "500M")
    env.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    # conda-forge layout on Windows: <prefix>/Library/bin/xtb.exe with params in <prefix>/Library/share/xtb
    for share in (binary.parent.parent / "share" / "xtb",):
        if share.is_dir():
            env.setdefault("XTBPATH", str(share))
    if sys.platform.startswith("win"):
        env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    return env


class XTBBinary(Calculator):
    """ASE calculator that shells out to ``xtb`` for each geometry."""

    implemented_properties = ["energy", "forces", "dipole", "charges"]
    name = "xtb"

    def __init__(
        self,
        binary: str | Path | None = None,
        method: str = "gfn2",
        charge: int = 0,
        uhf: int = 0,
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        timeout_s: float = 600.0,
        scratch_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        found = find_xtb_binary(str(binary) if binary else None)
        if found is None:
            raise FileNotFoundError(
                "xtb executable not found. Install it (conda-forge 'xtb' on Windows/Linux) and put it "
                "on PATH or set IR_AGENT_XTB_BIN; see README 'Windows install'."
            )
        self.binary = found
        self.method = method.lower().replace("-xtb", "").replace("gfn", "")
        if self.method not in ("0", "1", "2"):
            raise ValueError(f"Unknown xtb method {method!r}; use gfn0, gfn1 or gfn2")
        self.charge = int(charge)
        self.uhf = int(uhf)
        self.accuracy = accuracy
        self.etemp = electronic_temperature
        self.timeout_s = timeout_s
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None

    @property
    def method_label(self) -> str:
        return f"GFN{self.method}-xTB"

    def _args(self) -> list[str]:
        return [
            "--gfn", self.method,
            "--chrg", str(self.charge),
            "--uhf", str(self.uhf),
            "--acc", f"{self.accuracy}",
            "--etemp", f"{self.etemp}",
            "--norestart",
        ]

    def _run(self, atoms: Atoms, extra: list[str], workdir: Path) -> subprocess.CompletedProcess:
        if atoms.pbc.any():
            raise CalculatorError(
                "The xtb preset treats non-periodic systems only. Use a cluster model "
                "(surface.model='cluster') or a periodic calculator (uma, emt, tblite)."
            )
        write(workdir / "coord.xyz", atoms, format="xyz")
        cmd = [str(self.binary), "coord.xyz", *self._args(), *extra]
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=_subprocess_env(self.binary),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            text = proc.stdout + proc.stderr
            keys = ("did not converge", "cannot be satisfied", "[error]", "abnormal")
            causes = [ln.strip() for ln in text.splitlines() if any(k in ln.lower() for k in keys)]
            hint = ""
            if any("converge" in c.lower() or "satisfied" in c.lower() for c in causes):
                hint = (" SCC did not converge (seen for metallic Au clusters with GFN2-xTB; the "
                        "xtb-gfn1 preset converges there).")
            raise CalculatorError(
                f"xtb failed (exit {proc.returncode}).{hint} " + " | ".join(dict.fromkeys(causes))[:500]
            )
        return proc

    def _tmpdir(self) -> tempfile.TemporaryDirectory:
        if self.scratch_dir is not None:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="xtb_", dir=self.scratch_dir)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        with self._tmpdir() as tmp:
            wd = Path(tmp)
            self._run(self.atoms, ["--grad", "--json"], wd)
            energy, grad = _read_engrad(wd / "coord.engrad", len(self.atoms))
            js = _read_json(wd / "xtbout.json")
        self.results["energy"] = energy * Hartree
        self.results["free_energy"] = energy * Hartree
        self.results["forces"] = -grad * Hartree / Bohr
        # key is "dipole" in xtb 6.6 and "dipole / a.u." in 6.7; value in e*bohr
        dkey = next((k for k in js if k.split("/")[0].strip() == "dipole"), None)
        if dkey is None:
            raise CalculatorError("xtbout.json has no dipole entry")
        self.results["dipole"] = np.asarray(js[dkey], dtype=float) * Bohr  # e*bohr -> e*A
        if "partial charges" in js:
            self.results["charges"] = np.asarray(js["partial charges"], dtype=float)

    def native_ir(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        """Run ``xtb --hess`` and return (frequencies cm^-1, IR intensities km/mol).

        Rigid-body modes (reported as 0 by xtb) are removed.
        """
        freqs, ints, _ = self.hessian_run(atoms)
        keep = np.abs(freqs) > 1e-3
        return freqs[keep], ints[keep]

    def hessian(self, atoms: Atoms) -> np.ndarray:
        """Full Cartesian Hessian (3N x 3N, eV/A^2) from ``xtb --hess``.

        Use this instead of finite differences of ``--grad`` forces: with xtb 6.7.1 the single-point
        ``--grad`` gradient of slightly distorted, near-symmetric geometries (bent CO2, NH3
        deformations) is not rotationally invariant (torque ~1e-3 Eh on an isolated molecule) and
        disagrees with -dE/dx. Finite differences of it split degenerate modes (NH3 deg. deformation
        1236/1611 instead of 1611/1611 cm^-1 with GFN1). ``--hess`` agrees with ``--ohess`` and with
        dxtb's autograd GFN1 Hessian (tests/test_xtb_hessian.py).
        """
        return self.hessian_run(atoms)[2]

    def hessian_run(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._tmpdir() as tmp:
            wd = Path(tmp)
            self._run(atoms, ["--hess"], wd)
            freqs, ints = _read_vibspectrum(wd / "vibspectrum")
            hess = _read_hessian(wd / "hessian", len(atoms)) * Hartree / Bohr**2
        return freqs, ints, hess


def _read_json(path: Path) -> dict:
    """xtbout.json; on Windows the "program call" holds unescaped backslashes, so repair it."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # only the "program call" string contains backslashes (Windows paths)
        return json.loads(text.replace("\\", "/"))


def _read_engrad(path: Path, natoms: int) -> tuple[float, np.ndarray]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    data = [ln for ln in lines if ln and not ln.startswith("#")]
    n = int(data[0])
    if n != natoms:
        raise CalculatorError(f"engrad atom count {n} != {natoms}")
    energy = float(data[1])
    grad = np.array([float(x) for x in data[2 : 2 + 3 * n]]).reshape(n, 3)
    return energy, grad


def _read_hessian(path: Path, natoms: int) -> np.ndarray:
    """Turbomole-style ``$hessian`` file written by ``xtb --hess`` (Eh/bohr^2, 3N x 3N)."""
    vals: list[float] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("$"):
            continue
        vals.extend(float(x) for x in s.split())
    n3 = 3 * natoms
    if len(vals) != n3 * n3:
        raise CalculatorError(f"hessian file has {len(vals)} values, expected {n3 * n3}")
    h = np.array(vals).reshape(n3, n3)
    return 0.5 * (h + h.T)


def _read_vibspectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs, ints = [], []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith(("$", "#")):
            continue
        parts = s.split()
        # "mode  [symmetry]  wavenumber  intensity  selection  ..." ; symmetry may be absent
        nums = []
        for p in parts[1:]:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) >= 2:
            freqs.append(nums[0])
            ints.append(nums[1])
    return np.array(freqs), np.array(ints)
