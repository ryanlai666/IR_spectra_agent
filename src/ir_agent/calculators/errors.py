"""Calculator-related exceptions."""

from __future__ import annotations


class CalculatorUnavailableError(RuntimeError):
    """A requested calculator preset cannot be used here (not installed, no license/token, ...)."""

    def __init__(self, preset: str, reason: str):
        self.preset = preset
        self.reason = reason
        super().__init__(f"Calculator preset '{preset}' is unavailable: {reason}")


class UnsupportedSystemError(ValueError):
    """The preset exists but cannot treat this kind of system (e.g. xTB binary + PBC slab)."""
