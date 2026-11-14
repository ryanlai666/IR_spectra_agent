"""Calculator presets (EMT, xTB, dxtb, tblite, UMA, DFTB+, GPAW, NWChem, VASP)."""

from ir_agent.calculators.errors import CalculatorUnavailableError, UnsupportedSystemError
from ir_agent.calculators.registry import PRESETS, canonical_preset, list_presets, make_calculator

__all__ = [
    "PRESETS",
    "CalculatorUnavailableError",
    "UnsupportedSystemError",
    "canonical_preset",
    "list_presets",
    "make_calculator",
]
