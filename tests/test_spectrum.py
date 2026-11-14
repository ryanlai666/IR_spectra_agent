from __future__ import annotations

import numpy as np
import pytest

from ir_agent.spectrum import broaden, gaussian, lorentzian


@pytest.mark.parametrize("shape", [gaussian, lorentzian])
def test_lineshapes_are_area_normalized(shape):
    x = np.linspace(-20000, 20000, 400001)
    y = shape(x, 0.0, 20.0)
    area = np.trapezoid(y, x)
    assert area == pytest.approx(1.0, abs=2e-3 if shape is lorentzian else 1e-6)


@pytest.mark.parametrize("shape", [gaussian, lorentzian])
def test_fwhm(shape):
    x = np.linspace(-100, 100, 200001)
    y = shape(x, 0.0, 30.0)
    half = x[y >= y.max() / 2]
    assert half.max() - half.min() == pytest.approx(30.0, abs=0.01)


def test_broaden_integrates_to_total_intensity_and_peaks_at_sticks():
    spec = broaden(np.array([-30.0, 10.0, 1600.0, 3700.0]), np.array([5.0, 9.0, 60.0, 40.0]),
                   fwhm_cm=10.0, lineshape="gaussian", x_min_cm=0, x_max_cm=4500, n_points=45001)
    assert spec.sticks_cm.tolist() == [1600.0, 3700.0]  # imaginary and < 50 cm-1 dropped
    assert np.trapezoid(spec.y, spec.x_cm) == pytest.approx(100.0, rel=1e-3)
    peaks = [p for p, _ in spec.peaks()]
    assert peaks == pytest.approx([1600.0, 3700.0], abs=0.2)
    assert spec.y_units.startswith("km/mol")


def test_broaden_without_intensities_is_flagged():
    spec = broaden(np.array([1000.0]), None)
    assert "arb" in spec.y_units


def test_scale_factor_shifts_sticks():
    spec = broaden(np.array([1000.0]), np.array([1.0]), scale_factor=0.96)
    assert spec.sticks_cm[0] == pytest.approx(960.0)


def test_bad_inputs():
    with pytest.raises(ValueError):
        broaden(np.array([1000.0]), np.array([1.0]), fwhm_cm=0)
    with pytest.raises(ValueError):
        broaden(np.array([1000.0]), np.array([1.0]), lineshape="voigt")
