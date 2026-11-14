"""Line broadening of stick spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def lorentzian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Area-normalized Lorentzian."""
    g = 0.5 * fwhm
    return (g / np.pi) / ((x - center) ** 2 + g**2)


def gaussian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Area-normalized Gaussian."""
    s = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((x - center) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


_LINESHAPES = {"lorentzian": lorentzian, "gaussian": gaussian}


@dataclass
class Spectrum:
    x_cm: np.ndarray
    y: np.ndarray
    sticks_cm: np.ndarray
    stick_intensities: np.ndarray
    lineshape: str
    fwhm_cm: float
    scale_factor: float
    y_units: str  #: "km/mol per cm^-1" or "arb. units"

    def peaks(self, rel_height: float = 0.02) -> list[tuple[float, float]]:
        """Local maxima of the broadened curve above ``rel_height`` * max."""
        y = self.y
        if y.size < 3 or y.max() <= 0:
            return []
        idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
        thr = rel_height * y.max()
        return [(float(self.x_cm[i]), float(y[i])) for i in idx if y[i] >= thr]


def broaden(
    frequencies_cm: np.ndarray,
    intensities: np.ndarray | None,
    fwhm_cm: float = 20.0,
    lineshape: str = "lorentzian",
    x_min_cm: float = 400.0,
    x_max_cm: float = 4000.0,
    n_points: int = 3601,
    scale_factor: float = 1.0,
    min_frequency_cm: float = 50.0,
) -> Spectrum:
    """Broaden a stick spectrum.

    Imaginary/low modes (< ``min_frequency_cm``) are dropped. If ``intensities`` is None every
    mode gets unit height and ``y_units`` says so (frequency-only result, *not* an IR spectrum).
    Each line integrates to its stick intensity, so ``y`` is in km/mol per cm^-1.
    """
    if fwhm_cm <= 0:
        raise ValueError("fwhm_cm must be positive")
    if lineshape not in _LINESHAPES:
        raise ValueError(f"lineshape must be one of {sorted(_LINESHAPES)}")
    f = np.asarray(frequencies_cm, dtype=float)
    keep = f > min_frequency_cm
    sticks = f[keep] * scale_factor
    if intensities is None:
        ints = np.ones_like(sticks)
        units = "arb. units (no dipoles: all modes unit height)"
    else:
        ints = np.asarray(intensities, dtype=float)[keep]
        units = "km/mol per cm^-1"
    x = np.linspace(x_min_cm, x_max_cm, n_points)
    shape = _LINESHAPES[lineshape]
    y = np.zeros_like(x)
    for c, a in zip(sticks, ints, strict=True):
        y += a * shape(x, c, fwhm_cm)
    return Spectrum(x, y, sticks, ints, lineshape, fwhm_cm, scale_factor, units)
