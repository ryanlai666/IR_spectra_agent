"""ASE wrapper around dxtb (GFN1-xTB in PyTorch; CPU or CUDA, pip-installable on Windows).

* Energies/forces: autograd through the SCF (works everywhere).
* Dipoles: dxtb's field-derivative dipole needs multipole integrals from ``tad-libcint``
  (Linux only). Without it we fall back to the Mulliken-charge dipole mu = sum_i q_i(R) r_i.
  Finite differences of that expression keep the charge-flux term (dq/dR) but miss the
  atomic-polarization part, so intensities are approximate (flagged in ``dipole_model``).

dxtb works in atomic units (bohr, hartree, e*bohr).
"""

from __future__ import annotations

import importlib.util

import numpy as np
from ase.calculators.calculator import Calculator, CalculatorError, all_changes
from ase.units import Bohr, Hartree

from ir_agent.runtime import get_logger

log = get_logger("dxtb")

#: +1 if dxtb's atomic "charges" are partial charges (e), -1 if they are electron excesses.
_DXTB_CHARGE_SIGN = 1


def libcint_available() -> bool:
    return importlib.util.find_spec("tad_libcint") is not None


class DXTBCalculator(Calculator):
    implemented_properties = ["energy", "forces", "dipole", "charges"]
    name = "dxtb"

    def __init__(self, method: str = "gfn1", charge: int = 0, spin: int = 1, device: str = "cpu", **kw):
        super().__init__(**kw)
        import torch

        if method.lower() not in ("gfn1", "gfn2"):
            raise ValueError("dxtb method must be gfn1 or gfn2")
        self.full_dipole = libcint_available()
        if method.lower() == "gfn2" and not self.full_dipole:
            raise CalculatorError("dxtb GFN2-xTB needs tad-libcint (Linux only); use method gfn1 or the xtb preset")
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("dxtb: CUDA not available, using CPU")
            device = "cpu"
        from ir_agent.gpu import warn_if_unguarded

        warn_if_unguarded(device)
        self.method = method.lower()
        self.charge = int(charge)
        self.uhf = int(spin) - 1
        self.dd = {"device": torch.device(device), "dtype": torch.float64}
        self._cache: dict[tuple[int, ...], object] = {}
        self.dipole_model = "field derivative (libcint)" if self.full_dipole else "Mulliken charges (no libcint)"

    @property
    def method_label(self) -> str:
        return f"{self.method.upper()}-xTB (dxtb)"

    def _calc_for(self, numbers: np.ndarray):
        import dxtb
        import torch

        key = tuple(int(z) for z in numbers)
        if key not in self._cache:
            par = dxtb.GFN1_XTB if self.method == "gfn1" else dxtb.GFN2_XTB
            nums = torch.tensor(key, device=self.dd["device"])
            opts = {"verbosity": 0}
            if self.full_dipole:
                ef = dxtb.components.field.new_efield(torch.zeros(3, **self.dd))
                calc = dxtb.Calculator(nums, par, interaction=[ef], opts=opts, **self.dd)
            else:
                opts["int_driver"] = "dxtb"  # pure-PyTorch overlap integrals (no multipoles)
                calc = dxtb.Calculator(nums, par, opts=opts, **self.dd)
            self._cache[key] = calc
        return self._cache[key]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        import torch

        super().calculate(atoms, properties, system_changes)
        if self.atoms.pbc.any():
            raise CalculatorError("dxtb preset treats non-periodic systems only")
        calc = self._calc_for(self.atoms.get_atomic_numbers())
        pos = torch.tensor(self.atoms.get_positions() / Bohr, **self.dd)
        spin = self.uhf if self.uhf > 0 else None
        calc.reset()
        p = pos.clone().requires_grad_(True)
        energy = calc.get_energy(p, chrg=self.charge, spin=spin)
        (grad,) = torch.autograd.grad(energy, p)
        calc.reset()
        if self.full_dipole:
            dip = calc.get_dipole(pos.clone(), chrg=self.charge, spin=spin).detach().cpu().numpy().reshape(3)
            q = None
        else:
            q_ao = calc.get_charges(pos.clone(), chrg=self.charge, spin=spin)  # orbital-resolved
            q = calc.ihelp.reduce_orbital_to_atom(q_ao).detach().cpu().numpy()
            # Mulliken "charges" here are populations relative to the neutral reference
            # (positive = electron excess in dxtb's convention); sign checked against xtb in tests.
            dip = -q @ pos.cpu().numpy() if _DXTB_CHARGE_SIGN < 0 else q @ pos.cpu().numpy()
        fmax = float(grad.detach().abs().max()) * Hartree / Bohr
        if not np.isfinite(float(energy.detach())) or not np.isfinite(fmax) or fmax > 1e3:
            raise CalculatorError(
                f"dxtb SCF diverged (max |F| = {fmax:.3g} eV/A). The pure-PyTorch dxtb build is "
                "unreliable for transition-metal clusters; use the xtb or xtb-gfn1 presets there."
            )
        self.results["energy"] = float(energy.detach()) * Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -grad.detach().cpu().numpy() * Hartree / Bohr
        self.results["dipole"] = np.asarray(dip) * Bohr
        if q is not None:
            self.results["charges"] = q
