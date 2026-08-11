# SPDX-License-Identifier: MIT

"""Classes and functions for modelling recruitment to clinical trials.

Random Seed Strategy: {recruitment_seed_strategy}
"""

from __future__ import annotations

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import abc

from clintrials.core.recruitment_geometry import RecruitmentGeometry
from clintrials.core.recruitment_state import RecruitmentStreamState


class RecruitmentStream(metaclass=abc.ABCMeta):
    """Abstract base class for recruitment streams."""

    @abc.abstractmethod
    def reset(self):  # type: ignore
        """Resets the recruitment stream to its initial state."""
        pass

    @abc.abstractmethod
    def next(self):  # type: ignore
        """Gets the recruitment time of the next patient.

        Returns:
            float: The recruitment time of the next patient.
        """
        pass


class ConstantRecruitmentStream(RecruitmentStream):
    """A recruitment stream with a constant wait time between patients.

    This class models a simple recruitment scenario where a new patient arrives
    at regular intervals.

    Examples:
        >>> s = ConstantRecruitmentStream(2.5)
        >>> s.next()
        2.5
        >>> s.next()
        5.0
        >>> s.next()
        7.5
        >>> s.reset()
        >>> s.next()
        2.5
    """

    def __init__(self, intrapatient_gap):  # type: ignore
        """Initializes a ConstantRecruitmentStream object.

        Args:
            intrapatient_gap (float): The constant time gap between patient
                recruitments. Must be strictly positive.

        Raises:
            ValueError: If `intrapatient_gap` is not strictly positive.
        """
        if intrapatient_gap <= 0:
            raise ValueError("intrapatient_gap must be strictly positive.")
        self.delta = intrapatient_gap
        self.cursor = 0

    def reset(self):  # type: ignore
        """Resets the recruitment stream to its initial state."""
        self.cursor = 0

    def next(self):  # type: ignore
        """Gets the recruitment time of the next patient.

        Returns:
            float: The recruitment time of the next patient.
        """
        self.cursor += self.delta
        return self.cursor


class QuadrilateralRecruitmentStream(RecruitmentStream):
    """A recruitment stream with time-varying recruitment potential.

    This class models recruitment scenarios where the rate of patient arrival
    changes over time. The recruitment potential is defined by a series of
    vertices, and the intensity between these vertices can be either
    linearly interpolated or stepped.

    Examples:
        >>> s1 = QuadrilateralRecruitmentStream(4.0, 0.5, [(20, 1.0)], interpolate=True)
        >>> s1.next()
        6.832815729997477
        >>> s1.next()
        12.2490309931942
        >>> s2 = QuadrilateralRecruitmentStream(4.0, 0.5, [(20, 1.0)], interpolate=False)
        >>> s2.next()
        8.0
        >>> s2.next()
        16.0
    """

    def __init__(self, intrapatient_gap, initial_intensity, vertices, interpolate=True):  # type: ignore
        """Initializes a QuadrilateralRecruitmentStream object.

        Args:
            intrapatient_gap (float): The time to recruit one patient at 100%
                recruitment intensity. Must be strictly positive.
            initial_intensity (float): The initial recruitment intensity, as a
                proportion of total power. Must be non-negative. Zero is allowed
                to model delayed recruitment start.
            vertices (list[tuple[float, float]]): A list of (time, intensity)
                tuples representing vertices where the recruitment intensity
                changes.
            interpolate (bool, optional): Whether to linearly interpolate
                between vertices (`True`) or use stepped transitions (`False`).
                Defaults to `True`.

        Raises:
            ValueError: If `intrapatient_gap` is not strictly positive,
                if `initial_intensity` is negative, or if any of the
                intensities in `vertices` are negative.
        """
        self.geometry = RecruitmentGeometry(
            intrapatient_gap=intrapatient_gap,
            initial_intensity=initial_intensity,
            vertices=vertices,
            interpolate=interpolate,
        )
        self.state = RecruitmentStreamState(self.geometry)

    @property
    def delta(self) -> float:
        """The constant time gap between patient recruitments."""
        return self.geometry.delta

    @property
    def initial_intensity(self) -> float:
        """The initial recruitment intensity."""
        return self.geometry.initial_intensity

    @property
    def interpolate(self) -> bool:
        """Whether to linearly interpolate between vertices."""
        return self.geometry.interpolate

    @property
    def vertices(self) -> list[tuple[float, float]]:
        """The sorted vertices representing the recruitment intensity profile."""
        return self.geometry.vertices

    @property
    def shapes(self) -> dict[float, tuple[float, float, float, float]]:
        """The shape properties of the intervals."""
        return self.geometry.shapes

    @property
    def recruiment_mass(self) -> dict[float, float]:
        """The total recruitment potential mass of the intervals."""
        return self.geometry.recruiment_mass

    @property
    def cursor(self) -> float:
        """The current simulation time cursor of the stream."""
        return self.state.cursor

    @cursor.setter
    def cursor(self, value: float) -> None:
        self.state.cursor = value

    @property
    def available_mass(self) -> dict[float, float]:
        """The remaining recruitment potential mass in each interval."""
        return self.state.available_mass

    @available_mass.setter
    def available_mass(self, value: dict[float, float]) -> None:
        self.state.available_mass = value

    def reset(self) -> None:
        """Resets the recruitment stream to its initial state."""
        self.state.reset()

    def next(self) -> float:
        """Gets the recruitment time of the next patient.

        Returns:
            float: The recruitment time of the next patient.
        """
        return self.state.next_patient()

    def _linearly_interpolate_y(self, t, t0, t1, y0, y1):  # type: ignore
        """Linearly interpolates the y-value at time t.

        Args:
            t (float): The time at which to interpolate the y-value.
            t0 (float): The start time of the interval.
            t1 (float): The end time of the interval.
            y0 (float): The y-value at time t0.
            y1 (float): The y-value at time t1.

        Returns:
            float: The interpolated y-value at time t.
        """
        from clintrials.core.recruitment_solver import interpolate_intensity
        return interpolate_intensity(t, t0, t1, y0, y1)

    def _invert(self, t0, t1, y0, y1, mass, as_rectangle=False):  # type: ignore
        """Calculates the time at which the area under the curve equals a given mass.

        The area is calculated for a quadrilateral with vertices at t0, t, f(t),
        and f(t0), where f(t) is the recruitment intensity function.

        Args:
            t0 (float): The start time of the interval.
            t1 (float): The end time of the interval.
            y0 (float): The recruitment intensity at time t0.
            y1 (float): The recruitment intensity at time t1.
            mass (float): The target area (recruitment mass).
            as_rectangle (bool, optional): If `True`, treat the area as a
                rectangle. Defaults to `False`.

        Returns:
            float: The time `t` at which the cumulative recruitment mass
                equals the target `mass`.
        """
        from clintrials.core.recruitment_solver import invert_mass_to_time
        return invert_mass_to_time(t0, t1, y0, y1, mass, as_rectangle=as_rectangle)


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY

    __doc__ = __doc__.format(**CORE_REGISTRY)
