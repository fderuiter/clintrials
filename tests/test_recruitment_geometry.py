# SPDX-License-Identifier: MIT

"""Unit tests for RecruitmentGeometry class and validation checks."""

from __future__ import annotations

import pytest

from clintrials.core.recruitment_geometry import (
    RecruitmentGeometry,
    RecruitmentProfileGeometry,
)


def test_recruitment_geometry_alias() -> None:
    # Ensure both names reference the same class
    assert RecruitmentGeometry is RecruitmentProfileGeometry


def test_recruitment_geometry_valid_initialization() -> None:
    # Basic valid initialization
    geom = RecruitmentGeometry(
        intrapatient_gap=4.0,
        initial_intensity=0.5,
        vertices=[(20.0, 1.0)],
        interpolate=True,
    )
    assert geom.delta == 4.0
    assert geom.initial_intensity == 0.5
    assert geom.vertices == [(20.0, 1.0)]
    assert geom.interpolate is True

    # Pre-computed shapes
    # (t0, t1, y0, y1) -> shapes[20.0] should be (0.0, 20.0, 0.5, 1.0)
    assert geom.shapes[20.0] == (0.0, 20.0, 0.5, 1.0)

    # Pre-computed masses
    # For t0=0, t1=20, y0=0.5, y1=1.0, interpolate=True:
    # mass = 0.5 * 20 * (0.5 + 1.0) = 15.0
    assert geom.recruiment_mass[20.0] == 15.0


def test_recruitment_geometry_stepped_initialization() -> None:
    # Stepped transitions (interpolate=False)
    geom = RecruitmentGeometry(
        intrapatient_gap=4.0,
        initial_intensity=0.5,
        vertices=[(20.0, 1.0)],
        interpolate=False,
    )
    # mass = (20 - 0) * 0.5 = 10.0
    assert geom.recruiment_mass[20.0] == 10.0


def test_recruitment_geometry_sorting() -> None:
    # Vertices should be sorted by time
    geom = RecruitmentGeometry(
        intrapatient_gap=5.0,
        initial_intensity=0.1,
        vertices=[(90.0, 0.25), (180.0, 1.0), (150.0, 0.75)],
    )
    assert geom.vertices == [(90.0, 0.25), (150.0, 0.75), (180.0, 1.0)]


def test_recruitment_geometry_invalid_gap_raises_error() -> None:
    with pytest.raises(ValueError, match="intrapatient_gap must be strictly positive"):
        RecruitmentGeometry(intrapatient_gap=0.0, initial_intensity=1.0, vertices=[])

    with pytest.raises(ValueError, match="intrapatient_gap must be strictly positive"):
        RecruitmentGeometry(intrapatient_gap=-1.5, initial_intensity=1.0, vertices=[])


def test_recruitment_geometry_invalid_initial_intensity_raises_error() -> None:
    with pytest.raises(ValueError, match="initial_intensity must be non-negative"):
        RecruitmentGeometry(intrapatient_gap=1.0, initial_intensity=-0.1, vertices=[])


def test_recruitment_geometry_invalid_vertex_intensity_raises_error() -> None:
    with pytest.raises(ValueError, match="intensity in vertices cannot be negative"):
        RecruitmentGeometry(
            intrapatient_gap=1.0,
            initial_intensity=1.0,
            vertices=[(10.0, 1.0), (20.0, -0.5)],
        )


def test_recruitment_geometry_empty_vertices_is_allowed() -> None:
    # This should initialize successfully with empty shapes and masses
    geom = RecruitmentGeometry(
        intrapatient_gap=1.0, initial_intensity=1.0, vertices=[]
    )
    assert geom.vertices == []
    assert geom.shapes == {}
    assert geom.recruiment_mass == {}
