# SPDX-License-Identifier: MIT

"""Unit tests for stateless mathematical recruitment solver functions."""

from __future__ import annotations

import numpy as np
import pytest

from clintrials.core.recruitment_solver import (
    calculate_interval_mass,
    interpolate_intensity,
    invert_mass_to_time,
)


def test_interpolate_intensity_basic() -> None:
    # Midpoint interpolation
    val = interpolate_intensity(t=5.0, t0=0.0, t1=10.0, y0=1.0, y1=2.0)
    assert val == 1.5

    # Boundary interpolation
    assert interpolate_intensity(t=0.0, t0=0.0, t1=10.0, y0=1.0, y1=2.0) == 1.0
    assert interpolate_intensity(t=10.0, t0=0.0, t1=10.0, y0=1.0, y1=2.0) == 2.0


def test_interpolate_intensity_equal_time_returns_nan() -> None:
    # t0 == t1 must return np.nan
    val = interpolate_intensity(t=5.0, t0=5.0, t1=5.0, y0=1.0, y1=2.0)
    assert np.isnan(val)


def test_calculate_interval_mass_trapezoidal() -> None:
    # Trapezoidal: 0.5 * (10 - 0) * (1.0 + 2.0) = 15.0
    mass = calculate_interval_mass(t0=0.0, t1=10.0, y0=1.0, y1=2.0, interpolate=True)
    assert mass == 15.0


def test_calculate_interval_mass_rectangular() -> None:
    # Rectangular: (10 - 0) * 1.0 = 10.0
    mass = calculate_interval_mass(t0=0.0, t1=10.0, y0=1.0, y1=2.0, interpolate=False)
    assert mass == 10.0


def test_invert_mass_to_time_equal_time_returns_nan() -> None:
    val = invert_mass_to_time(t0=5.0, t1=5.0, y0=1.0, y1=2.0, mass=5.0)
    assert np.isnan(val)


def test_invert_mass_to_time_zero_or_negative_y_returns_nan() -> None:
    # y0 == y1 and y0 <= 0
    assert np.isnan(invert_mass_to_time(t0=0.0, t1=10.0, y0=0.0, y1=0.0, mass=5.0))
    assert np.isnan(invert_mass_to_time(t0=0.0, t1=10.0, y0=-1.0, y1=-1.0, mass=5.0))


def test_invert_mass_to_time_as_rectangle() -> None:
    # t0 + 1.0 * mass / y0 = 2.0 + 1.0 * 6.0 / 3.0 = 4.0
    val = invert_mass_to_time(
        t0=2.0, t1=10.0, y0=3.0, y1=5.0, mass=6.0, as_rectangle=True
    )
    assert val == 4.0


def test_invert_mass_to_time_equal_intensity_acts_as_rectangle() -> None:
    # y0 == y1 > 0
    # t0 + 1.0 * mass / y0 = 1.0 + 4.0 / 2.0 = 3.0
    val = invert_mass_to_time(t0=1.0, t1=10.0, y0=2.0, y1=2.0, mass=4.0)
    assert val == 3.0


def test_invert_mass_to_time_trapezoidal_success() -> None:
    # Let's verify a known trapezoidal inversion.
    # If t0=0, t1=10, y0=1, y1=3 (m = 0.2).
    # If mass = 0.5 * t * (y0 + y) = 0.5 * t * (1 + 1 + 0.2 * t) = t + 0.1 * t^2.
    # If we want t = 5, mass = 5 + 0.1 * 25 = 7.5.
    val = invert_mass_to_time(t0=0.0, t1=10.0, y0=1.0, y1=3.0, mass=7.5)
    assert pytest.approx(val) == 5.0


def test_invert_mass_to_time_negative_discriminant_raises_typeerror() -> None:
    # If m * mass is negative and large enough such that y0**2 + 2 * m * mass < 0.
    # For example, t0=0, t1=1, y0=1, y1=0 (m = -1).
    # y0**2 + 2 * m * mass = 1 + 2 * (-1) * 1 = -1 < 0.
    with pytest.raises(TypeError, match="Discriminant is negative"):
        invert_mass_to_time(t0=0.0, t1=1.0, y0=1.0, y1=0.0, mass=1.0)
