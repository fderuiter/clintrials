# SPDX-License-Identifier: MIT

"""Stream state tracking for clinical trial recruitment modeling."""

from __future__ import annotations

import copy

import numpy as np

from clintrials.core.recruitment_geometry import RecruitmentGeometry
from clintrials.core.recruitment_solver import (
    interpolate_intensity,
    invert_mass_to_time,
)


class RecruitmentStreamState:
    """Manages the active state of a recruitment stream execution."""

    def __init__(
        self, geometry: RecruitmentGeometry, intrapatient_gap: float | None = None
    ) -> None:
        """Initializes the recruitment stream state.

        Args:
            geometry (RecruitmentGeometry): The configuration geometry for the profile.
            intrapatient_gap (float, optional): Custom time gap between patient recruitments.
                If not specified, defaults to the gap defined in the geometry.
        """
        self.geometry = geometry
        self.delta = intrapatient_gap if intrapatient_gap is not None else geometry.delta
        self.cursor = 0.0
        self.available_mass: dict[float, float] = {}
        self.reset()

    def reset(self) -> None:
        """Resets the recruitment stream state to its initial values."""
        self.cursor = 0.0
        self.available_mass = copy.copy(self.geometry.recruiment_mass)

    def next_patient(self) -> float:
        """Computes the arrival time of the next patient and advances the stream cursor.

        Returns:
            float: The recruitment time of the next patient.
        """
        sought_mass = self.delta
        t = sorted(self.available_mass.keys())
        for t1 in t:
            avail_mass = self.available_mass[t1]
            t0, _, y0, y1 = self.geometry.shapes[t1]
            if avail_mass >= sought_mass:
                if self.geometry.interpolate:
                    y_at_cursor = interpolate_intensity(
                        self.cursor, t0, t1, y0, y1
                    )
                    new_cursor = invert_mass_to_time(
                        self.cursor, t1, y_at_cursor, y1, sought_mass
                    )
                    self.cursor = new_cursor
                else:
                    y_at_cursor = y0
                    new_cursor = invert_mass_to_time(
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

        # satisfy outstanding sought mass using terminal recruitment intensity
        if len(self.geometry.vertices):
            _, y1 = self.geometry.vertices[-1]
            terminal_rate = y1
        else:
            terminal_rate = self.geometry.initial_intensity

        if terminal_rate > 0:
            self.cursor += sought_mass / terminal_rate
            return self.cursor
        else:
            return np.nan
