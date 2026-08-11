# SPDX-License-Identifier: MIT

"""Mathematical solver for clinical trial recruitment modeling."""

from __future__ import annotations

import numpy as np


def interpolate_intensity(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    """Linearly interpolates the recruitment intensity at time t.

    Args:
        t (float): The time at which to interpolate the intensity.
        t0 (float): The start time of the interval.
        t1 (float): The end time of the interval.
        y0 (float): The intensity at time t0.
        y1 (float): The intensity at time t1.

    Returns:
        float: The interpolated intensity, or np.nan if t1 == t0.
    """
    if t1 == t0:
        return np.nan
    m = (y1 - y0) / (t1 - t0)
    return y0 + m * (t - t0)


def invert_mass_to_time(
    t0: float, t1: float, y0: float, y1: float, mass: float, as_rectangle: bool = False
) -> float:
    """Calculates the time t at which the integrated area under the curve equals a given mass.

    Args:
        t0 (float): The start time of the interval.
        t1 (float): The end time of the interval.
        y0 (float): The recruitment intensity at time t0.
        y1 (float): The recruitment intensity at time t1.
        mass (float): The target recruitment mass.
        as_rectangle (bool, optional): If True, treat the area as a rectangle. Defaults to False.

    Returns:
        float: The calculated time t, or np.nan if t1 == t0 or (y0 == y1 and y0 <= 0).
    """
    if t1 == t0:
        return np.nan
    elif y0 == y1 and y0 <= 0:
        return np.nan
    elif (y0 == y1 and y0 > 0) or as_rectangle:
        return t0 + 1.0 * mass / y0
    else:
        m = (y1 - y0) / (t1 - t0)
        discriminant = y0**2 + 2 * m * mass
        if discriminant < 0:
            raise TypeError("Discriminant is negative")
        z = float(np.sqrt(discriminant))
        tau0 = (-y0 + z) / m
        tau1 = (-y0 - z) / m
        if tau0 + t0 > 0:
            return float(t0 + tau0)
        else:
            assert t0 + tau1 > 0
            return float(t0 + tau1)


def calculate_interval_mass(
    t0: float, t1: float, y0: float, y1: float, interpolate: bool
) -> float:
    """Calculates the integrated mass of an interval.

    Args:
        t0 (float): The start time of the interval.
        t1 (float): The end time of the interval.
        y0 (float): The intensity at time t0.
        y1 (float): The intensity at time t1.
        interpolate (bool): Whether to use trapezoidal interpolation (True) or rectangular step (False).

    Returns:
        float: The integrated mass of the interval.
    """
    if interpolate:
        return 0.5 * (t1 - t0) * (y0 + y1)
    else:
        return (t1 - t0) * y0
