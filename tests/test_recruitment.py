__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from clintrials.core.recruitment import (
    ConstantRecruitmentStream,
    QuadrilateralRecruitmentStream,
)


def test_constant_recruitment_stream():  # type: ignore

    s = ConstantRecruitmentStream(2)  # type: ignore

    assert s.next() == 2  # type: ignore
    assert s.next() == 4  # type: ignore
    assert s.next() == 6  # type: ignore
    s.reset()  # type: ignore
    assert s.next() == 2  # type: ignore


def test_quadrilateral_recruitment_stream_1():  # type: ignore

    initial = 1.0
    vertices = [(90, 1.0)]
    s = QuadrilateralRecruitmentStream(15.0, initial, vertices)  # type: ignore

    assert s.next() == 15.0  # type: ignore
    assert s.next() == 30.0  # type: ignore
    assert s.next() == 45.0  # type: ignore
    assert s.next() == 60.0  # type: ignore
    assert s.next() == 75.0  # type: ignore
    assert s.next() == 90.0  # type: ignore
    s.reset()  # type: ignore
    assert s.next() == 15.0  # type: ignore


def test_quadrilateral_recruitment_stream_2():  # type: ignore

    initial = 1.0
    vertices = [(90, 1.0)]
    s = QuadrilateralRecruitmentStream(15.0, initial, vertices, interpolate=False)  # type: ignore

    assert s.next() == 15.0  # type: ignore
    assert s.next() == 30.0  # type: ignore
    assert s.next() == 45.0  # type: ignore
    assert s.next() == 60.0  # type: ignore
    assert s.next() == 75.0  # type: ignore
    assert s.next() == 90.0  # type: ignore
    s.reset()  # type: ignore
    assert s.next() == 15.0  # type: ignore


def test_quadrilateral_recruitment_stream_3():  # type: ignore

    initial = 0.5
    vertices = []  # type: ignore
    s = QuadrilateralRecruitmentStream(10, initial, vertices)  # type: ignore

    assert s.next() == 20.0  # type: ignore
    assert s.next() == 40.0  # type: ignore
    assert s.next() == 60.0  # type: ignore
    s.reset()  # type: ignore
    assert s.next() == 20.0  # type: ignore


def test_quadrilateral_recruitment_stream_4():  # type: ignore

    initial = 0.5
    vertices = []  # type: ignore
    s = QuadrilateralRecruitmentStream(10, initial, vertices, interpolate=False)  # type: ignore

    assert s.next() == 20.0  # type: ignore
    assert s.next() == 40.0  # type: ignore
    assert s.next() == 60.0  # type: ignore
    s.reset()  # type: ignore
    assert s.next() == 20.0  # type: ignore


def test_quadrilateral_recruitment_stream_5():  # type: ignore

    initial = 0.1
    vertices = [(90, 0.25), (180, 1), (150, 0.75)]
    s = QuadrilateralRecruitmentStream(5, initial, vertices)  # type: ignore

    assert_almost_equal(s.next(), 37.979589711327129)  # type: ignore
    assert_almost_equal(s.next(), 64.899959967967959)  # type: ignore
    assert_almost_equal(s.next(), 86.969384566990684)  # type: ignore
    assert_almost_equal(s.next(), 103.8178046004133)  # type: ignore
    assert_almost_equal(s.next(), 115.85696017507577)  # type: ignore
    assert_almost_equal(s.next(), 125.72670690061994)  # type: ignore
    assert_almost_equal(s.next(), 134.29670248402687)  # type: ignore
    assert_almost_equal(s.next(), 141.9756061276768)  # type: ignore
    assert_almost_equal(s.next(), 148.99438184514796)  # type: ignore
    assert_almost_equal(s.next(), 155.49869109050658)  # type: ignore
    assert_almost_equal(s.next(), 161.58740079360237)  # type: ignore
    assert_almost_equal(s.next(), 167.33126291998994)  # type: ignore
    assert_almost_equal(s.next(), 172.78297743897352)  # type: ignore
    assert_almost_equal(s.next(), 177.98304963002104)  # type: ignore
    assert_almost_equal(s.next(), 183.0)  # type: ignore
    assert_almost_equal(s.next(), 188.0)  # type: ignore
    s.reset()  # type: ignore
    assert_almost_equal(s.next(), 37.979589711327129)  # type: ignore


def test_quadrilateral_recruitment_stream_6():  # type: ignore

    initial = 0.1
    vertices = [(90, 0.25), (180, 1), (150, 0.75)]
    s = QuadrilateralRecruitmentStream(5, initial, vertices, interpolate=False)  # type: ignore

    assert_almost_equal(s.next(), 50.0)  # type: ignore
    assert_almost_equal(s.next(), 94.0)  # type: ignore
    assert_almost_equal(s.next(), 114.0)  # type: ignore
    assert_almost_equal(s.next(), 134.0)  # type: ignore
    assert_almost_equal(s.next(), 151.33333333333334)  # type: ignore
    assert_almost_equal(s.next(), 158.0)  # type: ignore
    assert_almost_equal(s.next(), 164.66666666666666)  # type: ignore
    assert_almost_equal(s.next(), 171.33333333333331)  # type: ignore
    assert_almost_equal(s.next(), 177.99999999999997)  # type: ignore
    assert_almost_equal(s.next(), 183.5)  # type: ignore
    assert_almost_equal(s.next(), 188.5)  # type: ignore
    s.reset()  # type: ignore
    assert_almost_equal(s.next(), 50.0)  # type: ignore


def test_quadrilateral_recruitment_stream_7():  # type: ignore

    initial = 0.0
    vertices = [(100, 1.0)]
    s = QuadrilateralRecruitmentStream(10.0, initial, vertices)  # type: ignore

    assert_almost_equal(s.next(), 44.721359549995789)  # type: ignore
    assert_almost_equal(s.next(), 63.245553203367578)  # type: ignore
    assert_almost_equal(s.next(), 77.459666924148337)  # type: ignore
    assert_almost_equal(s.next(), 89.442719099991578)  # type: ignore
    assert_almost_equal(s.next(), 99.999999999999986)  # type: ignore
    assert_almost_equal(s.next(), 110.0)  # type: ignore
    assert_almost_equal(s.next(), 120.0)  # type: ignore
    s.reset()  # type: ignore
    assert_almost_equal(s.next(), 44.721359549995789)  # type: ignore


def test_quadrilateral_recruitment_stream_8():  # type: ignore

    initial = 0.0
    vertices = [(100, 1.0), (130, 0.0), (150, 0.5)]
    s = QuadrilateralRecruitmentStream(10.0, initial, vertices, interpolate=False)  # type: ignore

    assert_almost_equal(s.next(), 110.0)  # type: ignore
    assert_almost_equal(s.next(), 120.0)  # type: ignore
    assert_almost_equal(s.next(), 130.0)  # type: ignore
    assert_almost_equal(s.next(), 170.0)  # type: ignore
    assert_almost_equal(s.next(), 190.0)  # type: ignore
    s.reset()  # type: ignore
    assert_almost_equal(s.next(), 110.0)  # type: ignore


def test_quadrilateral_recruitment_stream_9():  # type: ignore

    initial = 0.0
    vertices = [(100, 0.0), (200, 1.0), (250, 0.5)]
    s = QuadrilateralRecruitmentStream(10.0, initial, vertices)  # type: ignore

    assert_almost_equal(s.next(), 144.72135954999578)  # type: ignore
    assert_almost_equal(s.next(), 163.24555320336759)  # type: ignore
    assert_almost_equal(s.next(), 177.45966692414834)  # type: ignore
    assert_almost_equal(s.next(), 189.44271909999159)  # type: ignore
    assert_almost_equal(s.next(), 200.0)  # type: ignore
    assert_almost_equal(s.next(), 210.55728090000841)  # type: ignore
    assert_almost_equal(s.next(), 222.54033307585166)  # type: ignore
    assert_almost_equal(s.next(), 236.75444679663241)  # type: ignore
    assert_almost_equal(s.next(), 255.0)  # type: ignore
    assert_almost_equal(s.next(), 275.0)  # type: ignore
    s.reset()  # type: ignore
    assert_almost_equal(s.next(), 144.72135954999578)  # type: ignore


def test_linearly_interpolate_y_when_t1_equals_t0_returns_nan():  # type: ignore
    s = QuadrilateralRecruitmentStream(1.0, 1.0, [])  # type: ignore
    assert np.isnan(s._linearly_interpolate_y(1, 0, 0, 1, 1))  # type: ignore


def test_invert_negative_discriminant_raises_typeerror():  # type: ignore
    s = QuadrilateralRecruitmentStream(1.0, 1.0, [])  # type: ignore
    with pytest.raises(TypeError):
        s._invert(0, 1, 1, 0, 1)  # type: ignore


def test_constant_recruitment_stream_invalid_gap_raises_error():  # type: ignore
    with pytest.raises(ValueError, match="intrapatient_gap must be strictly positive"):
        ConstantRecruitmentStream(0)  # type: ignore
    with pytest.raises(ValueError, match="intrapatient_gap must be strictly positive"):
        ConstantRecruitmentStream(-1)  # type: ignore


def test_quadrilateral_recruitment_stream_invalid_gap_raises_error():  # type: ignore
    with pytest.raises(ValueError, match="intrapatient_gap must be strictly positive"):
        QuadrilateralRecruitmentStream(0, 1, [])  # type: ignore
    with pytest.raises(ValueError, match="intrapatient_gap must be strictly positive"):
        QuadrilateralRecruitmentStream(-1, 1, [])  # type: ignore


def test_quadrilateral_recruitment_stream_negative_initial_intensity_raises_error():  # type: ignore
    with pytest.raises(ValueError, match="initial_intensity must be non-negative"):
        QuadrilateralRecruitmentStream(1, -0.1, [])  # type: ignore


def test_quadrilateral_recruitment_stream_zero_initial_intensity_is_allowed():  # type: ignore
    # This should not raise an error
    QuadrilateralRecruitmentStream(1, 0, [])  # type: ignore
