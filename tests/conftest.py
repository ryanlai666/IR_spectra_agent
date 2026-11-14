from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ir_agent.config import AppConfig, load_config


class HarmonicDipoleCalc(Calculator):
    """Toy calculator: harmonic springs between listed pairs + fixed point-charge dipole.

    Energy E = sum_k 1/2 k (|r_i - r_j| - r0)^2 ; dipole mu = sum_i q_i r_i (e*A).
    Central-force springs are rotationally invariant, so a free diatomic has exactly one
    non-zero mode with omega = sqrt(k / mu_red).
    """

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, bonds, charges, **kw):
        super().__init__(**kw)
        self.bonds = bonds  # list of (i, j, k [eV/A^2], r0 [A])
        self.q = np.asarray(charges, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        e = 0.0
        f = np.zeros_like(pos)
        for i, j, k, r0 in self.bonds:
            d = pos[j] - pos[i]
            r = np.linalg.norm(d)
            e += 0.5 * k * (r - r0) ** 2
            g = k * (r - r0) * d / r  # dE/dr_j
            f[j] -= g
            f[i] += g
        self.results = {"energy": e, "free_energy": e, "forces": f, "dipole": self.q @ pos}


@pytest.fixture
def diatomic() -> tuple[Atoms, HarmonicDipoleCalc, dict]:
    params = {"k": 20.0, "r0": 1.13, "q": 0.3}
    atoms = Atoms("CO", positions=[[0, 0, 0], [0, 0, params["r0"]]], pbc=False)
    atoms.center(vacuum=5.0)
    calc = HarmonicDipoleCalc([(0, 1, params["k"], params["r0"])], [params["q"], -params["q"]])
    return atoms, calc, params


@pytest.fixture
def triatomic() -> tuple[Atoms, HarmonicDipoleCalc]:
    """Bent 'water-like' toy with three springs (two stretches + an H-H spring for the bend)."""
    atoms = Atoms("OH2", positions=[[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]], pbc=False)
    r_oh = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    r_hh = 1.52
    calc = HarmonicDipoleCalc(
        [(0, 1, 50.0, r_oh), (0, 2, 50.0, r_oh), (1, 2, 5.0, r_hh)], [-0.6, 0.3, 0.3]
    )
    return atoms, calc


@pytest.fixture
def cfg(tmp_path) -> AppConfig:
    return load_config(
        overrides={"run": {"output_root": str(tmp_path / "runs")},
                   "tracking": {"mlflow": False}},
        use_env=False,
    )
