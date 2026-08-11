# SPDX-License-Identifier: MIT

"""Unit tests for RecruitmentStreamState class."""

from __future__ import annotations

import numpy as np
import pytest

from clintrials.core.recruitment_geometry import RecruitmentGeometry
from clintrials.core.recruitment_state import RecruitmentStreamState


def test_recruitment_stream_state_initialization() -> None:
    # Set up a geometry with 1 interval
    geom = RecruitmentGeometry(
        intrapatient_gap=4.0,
        initial_intensity=1.0,
        vertices=[(10.0, 2.0)],
        interpolate=True,
    )

    # Initialize state with default gap
    state = RecruitmentStreamState(geom)
    assert state.geometry is geom
    assert state.delta == 4.0
    assert state.cursor == 0.0
    assert state.available_mass == {10.0: 15.0}  # 0.5 * 10.0 * (1.0 + 2.0) = 15.0

    # Initialize state with custom gap
    state_custom = RecruitmentStreamState(geom, intrapatient_gap=5.0)
    assert state_custom.delta == 5.0
    assert state_custom.cursor == 0.0


def test_recruitment_stream_state_reset() -> None:
    geom = RecruitmentGeometry(
        intrapatient_gap=4.0,
        initial_intensity=1.0,
        vertices=[(10.0, 2.0)],
        interpolate=True,
    )
    state = RecruitmentStreamState(geom)
    state.cursor = 5.5
    state.available_mass[10.0] = 5.0

    state.reset()
    assert state.cursor == 0.0
    assert state.available_mass == {10.0: 15.0}


def test_recruitment_stream_state_next_patient_constant() -> None:
    # Constant recruitment (no vertices)
    geom = RecruitmentGeometry(
        intrapatient_gap=2.0,
        initial_intensity=0.5,
        vertices=[],
    )
    state = RecruitmentStreamState(geom)
    # arrival time should be 2.0 / 0.5 = 4.0
    arrival1 = state.next_patient()
    assert arrival1 == 4.0
    assert state.cursor == 4.0

    # arrival time of next patient should be 4.0 + (2.0 / 0.5) = 8.0
    arrival2 = state.next_patient()
    assert arrival2 == 8.0
    assert state.cursor == 8.0


def test_recruitment_stream_state_next_patient_multi_interval() -> None:
    geom = RecruitmentGeometry(
        intrapatient_gap=5.0,
        initial_intensity=0.1,
        vertices=[(90.0, 0.25), (150.0, 0.75), (180.0, 1.0)],
        interpolate=True,
    )
    state = RecruitmentStreamState(geom)

    # The first patient arrivals should match the mathematics in original tests
    # which we can verify.
    arrival1 = state.next_patient()
    assert pytest.approx(arrival1) == 37.979589711327129
    assert pytest.approx(state.cursor) == 37.979589711327129


def test_recruitment_stream_state_next_patient_stepped() -> None:
    geom = RecruitmentGeometry(
        intrapatient_gap=5.0,
        initial_intensity=0.1,
        vertices=[(90.0, 0.25), (150.0, 0.75), (180.0, 1.0)],
        interpolate=False,
    )
    state = RecruitmentStreamState(geom)

    arrival1 = state.next_patient()
    assert pytest.approx(arrival1) == 50.0
    assert pytest.approx(state.cursor) == 50.0


def test_recruitment_stream_state_next_patient_zero_terminal_rate() -> None:
    # Geometry with zero initial intensity and no vertices
    geom = RecruitmentGeometry(
        intrapatient_gap=2.0,
        initial_intensity=0.0,
        vertices=[],
    )
    state = RecruitmentStreamState(geom)
    # Since terminal rate is 0 and no mass is available, it should return np.nan
    arrival = state.next_patient()
    assert np.isnan(arrival)
