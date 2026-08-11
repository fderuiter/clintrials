# SPDX-License-Identifier: MIT

"""Geometry configuration for clinical trial recruitment modeling."""

from __future__ import annotations

from clintrials.core.recruitment_solver import calculate_interval_mass


class RecruitmentGeometry:
    """Encapsulates and validates the geometric properties of a recruitment profile."""

    def __init__(
        self,
        intrapatient_gap: float,
        initial_intensity: float,
        vertices: list[tuple[float, float]],
        interpolate: bool = True,
    ) -> None:
        """Initializes and validates the recruitment geometry.

        Args:
            intrapatient_gap (float): The time to recruit one patient at 100% intensity.
                Must be strictly positive.
            initial_intensity (float): The initial recruitment intensity.
                Must be non-negative.
            vertices (list[tuple[float, float]]): Vertices where recruitment intensity
                changes, as (time, intensity) tuples.
            interpolate (bool, optional): Whether to interpolate linearly (True) or
                use stepped transitions (False). Defaults to True.

        Raises:
            ValueError: If validation of gap, initial intensity, or vertex intensities fails.
        """
        if intrapatient_gap <= 0:
            raise ValueError("intrapatient_gap must be strictly positive.")
        if initial_intensity < 0:
            raise ValueError(
                "initial_intensity must be non-negative. Zero is allowed to model "
                "delayed recruitment start."
            )
        if any(v[1] < 0 for v in vertices):
            raise ValueError("intensity in vertices cannot be negative.")

        self.delta = intrapatient_gap
        self.initial_intensity = initial_intensity
        self.interpolate = interpolate

        self.vertices = sorted(vertices, key=lambda x: x[0])
        self.shapes: dict[float, tuple[float, float, float, float]] = {}
        self.recruiment_mass: dict[float, float] = {}

        if len(self.vertices) > 0:
            t0 = 0.0
            y0 = initial_intensity
            for x in self.vertices:
                t1, y1 = x
                mass = calculate_interval_mass(t0, t1, y0, y1, interpolate)
                self.recruiment_mass[t1] = mass
                self.shapes[t1] = (t0, t1, y0, y1)
                t0, y0 = t1, y1


RecruitmentProfileGeometry = RecruitmentGeometry
