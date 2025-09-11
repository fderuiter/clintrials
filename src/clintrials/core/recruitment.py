__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import abc
import copy

import numpy as np

""" Classes and functions for modelling recruitment to clinical trials. """


class RecruitmentStream(metaclass=abc.ABCMeta):
    """Abstract base class for recruitment streams."""

    @abc.abstractmethod
    def reset(self):
        """Resets the recruitment stream to its initial state."""
        pass

    @abc.abstractmethod
    def next(self):
        """Gets the recruitment time of the next patient.

        Returns:
            The recruitment time of the next patient.
        """
        pass


class ConstantRecruitmentStream(RecruitmentStream):
    """A recruitment stream with a constant wait time between patients.

    In this simple case, a patient arrives every `delta` units of time.

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

    def __init__(self, intrapatient_gap):
        """Initializes the ConstantRecruitmentStream.

        Args:
            intrapatient_gap: The time gap between patient recruitments.
        """
        self.delta = intrapatient_gap
        self.cursor = 0

    def reset(self):
        """Resets the recruitment stream to its initial state."""
        self.cursor = 0

    def next(self):
        """Gets the recruitment time of the next patient.

        Returns:
            The recruitment time of the next patient.
        """
        self.cursor += self.delta
        return self.cursor


class QuadrilateralRecruitmentStream(RecruitmentStream):
    """A recruitment stream where recruitment intensity varies over time.

    Recruitment intensity is defined by a series of vertices, and can change
    linearly (interpolation) or instantaneously (steps) between them.

    Examples:
        >>> s1 = QuadrilateralRecruitmentStream(4.0, 0.5, [(20, 1.0)], interpolate=True)
        >>> s1.next()
        6.8328157299974768
        >>> s1.next()
        12.2490309931942
        >>> s1.next()
        16.878177829171548
        >>> s1.next()
        21.0
        >>> s1.next()
        25.0

        >>> s2 = QuadrilateralRecruitmentStream(4.0, 0.5, [(20, 1.0)], interpolate=False)
        >>> s2.next()
        8.0
        >>> s2.next()
        16.0
        >>> s2.next()
        22.0
        >>> s2.next()
        26.0
    """

    def __init__(self, intrapatient_gap, initial_intensity, vertices, interpolate=True):
        """Initializes the QuadrilateralRecruitmentStream.

        Args:
            intrapatient_gap: The time to recruit one patient at 100%
                recruitment intensity.
            initial_intensity: The initial recruitment intensity as a
                percentage of total power. Must be non-negative.
            vertices: A list of (time, intensity) tuples representing
                additional vertices.
            interpolate: If True, linearly interpolate between vertices;
                otherwise, use steps.

        Raises:
            ValueError: If `initial_intensity` or any intensity in `vertices`
                is negative.
        """
        if initial_intensity < 0:
            raise ValueError("initial_intensity cannot be negative.")
        if any(v[1] < 0 for v in vertices):
            raise ValueError("intensity in vertices cannot be negative.")

        self.delta = intrapatient_gap
        self.initial_intensity = initial_intensity
        self.interpolate = interpolate

        v = vertices
        v.sort(key=lambda x: x[0])
        self.shapes = {}  # t1 -> t0, t1, y0, y1 vertex parameters
        self.recruiment_mass = (
            {}
        )  # t1 -> recruitment mass available (i.e. area of quadrilateral) to left of t1
        if len(v) > 0:
            t0 = 0
            y0 = initial_intensity
            for x in v:
                t1, y1 = x
                if interpolate:
                    mass = 0.5 * (t1 - t0) * (y0 + y1)  # Area of trapezium
                else:
                    mass = (t1 - t0) * y0  # Are of rectangle
                self.recruiment_mass[t1] = mass
                self.shapes[t1] = (t0, t1, y0, y1)
                t0, y0 = t1, y1
            self.available_mass = copy.copy(self.recruiment_mass)
        else:
            self.available_mass = {}
        self.vertices = v
        self.cursor = 0

    def reset(self):
        """Reset the recruitment stream to start anew.

        :return: None
        :rtype: None

        """

        self.cursor = 0
        self.available_mass = copy.copy(self.recruiment_mass)

    def next(self):
        """Get the time that the next patient is recruited.

        :return: The time that the next patient is recruited.
        :rtype: float

        """

        sought_mass = self.delta
        t = sorted(self.available_mass.keys())
        for t1 in t:
            avail_mass = self.available_mass[t1]
            t0, _, y0, y1 = self.shapes[t1]
            if avail_mass >= sought_mass:
                if self.interpolate:
                    y_at_cursor = self._linearly_interpolate_y(
                        self.cursor, t0, t1, y0, y1
                    )
                    new_cursor = self._invert(
                        self.cursor, t1, y_at_cursor, y1, sought_mass
                    )
                    self.cursor = new_cursor
                else:
                    y_at_cursor = y0
                    new_cursor = self._invert(
                        self.cursor, t1, y_at_cursor, y1, sought_mass, as_rectangle=True
                    )
                    self.cursor = new_cursor

                self.available_mass[t1] -= sought_mass
                return self.cursor
            else:
                sought_mass -= avail_mass
                self.available_mass[t1] = 0.0
                if t1 > self.cursor:
                    self.cursor = t1

        # Got here? Satisfy outstanding sought mass using terminal recruitment intensity
        terminal_rate = y1 if len(self.vertices) else self.initial_intensity
        if terminal_rate > 0:
            self.cursor += sought_mass / terminal_rate
            return self.cursor
        else:
            return np.nan

    def _linearly_interpolate_y(self, t, t0, t1, y0, y1):
        """Linearly interpolate y-value at t using line through (t0, y0) and (t1, y1)"""
        if t1 == t0:
            # The line either has infiniite gradient or is not a line at all, but a point. No logical response
            return np.nan
        else:
            m = (y1 - y0) / (t1 - t0)
            return y0 + m * (t - t0)

    def _invert(self, t0, t1, y0, y1, mass, as_rectangle=False):
        """Returns time t at which the area of quadrilateral with vertices at t0, t, f(t), f(t0) equals mass."""
        if t1 == t0:
            # The quadrilateral has no area
            return np.nan
        elif y0 == y1 and y0 <= 0:
            # The quadrilateral has no area or is badly defined
            return np.nan
        elif (y0 == y1 and y0 > 0) or as_rectangle:
            # We require area of a rectangle; easy!
            return t0 + 1.0 * mass / y0
        else:
            # We require area of a trapezium. That requires solving a quadratic.
            m = (y1 - y0) / (t1 - t0)
            discriminant = y0**2 + 2 * m * mass
            if discriminant < 0:
                raise TypeError("Discriminant is negative")
            z = np.sqrt(discriminant)
            tau0 = (-y0 + z) / m
            tau1 = (-y0 - z) / m
            if tau0 + t0 > 0:
                return t0 + tau0
            else:
                assert t0 + tau1 > 0
                return t0 + tau1
