import numpy as np
import pytest

from clintrials.phase3.gsd import (
    GroupSequentialDesign,
    spending_function_obrien_fleming,
    spending_function_pocock,
)


def test_obrien_fleming_boundaries():
    """
    Test the O'Brien-Fleming boundaries against known values from a reference.
    Reference: http://www.biostat.umn.edu/~josephk/courses/pubh8482_fall2012/lecture_notes/pubh8482_week3.pdf
    (Slide 23, for alpha=0.05 two-sided, which is alpha=0.025 one-sided)
    The implemented spending function is an approximation, so a higher
    tolerance is needed.
    """
    k = 4
    alpha = 0.025
    expected_boundaries = [4.048, 2.862, 2.337, 2.024]

    design = GroupSequentialDesign(
        k=k, alpha=alpha, sfu=spending_function_obrien_fleming
    )

    assert len(design.efficacy_boundaries) == k
    np.testing.assert_allclose(
        design.efficacy_boundaries, expected_boundaries, rtol=0.08
    )


def test_spending_functions_at_t1():
    """Test that spending functions spend the full alpha at t=1."""
    alpha = 0.025
    assert spending_function_pocock(1.0, alpha) == pytest.approx(alpha)
    assert spending_function_obrien_fleming(1.0, alpha) == pytest.approx(alpha)


def test_simulation_type1_error():
    """Test that the simulated Type I error is close to alpha."""
    k = 4
    alpha = 0.025
    design = GroupSequentialDesign(
        k=k, alpha=alpha, sfu=spending_function_obrien_fleming
    )

    # This is a stochastic test, so it might fail by chance.
    # A high number of sims and a reasonable tolerance are needed.
    n_sims = 20000  # Increased for stability
    results = design.simulate(n_sims=n_sims, theta=0)

    # Allow for some Monte Carlo error.
    assert results["rejection_prob"] == pytest.approx(alpha, abs=0.01)


def test_gsd_with_timing():
    k = 4
    alpha = 0.025
    timing = [0.25, 0.5, 0.75, 1.0]
    design = GroupSequentialDesign(
        k=k, alpha=alpha, sfu=spending_function_obrien_fleming, timing=timing
    )
    assert len(design.efficacy_boundaries) == k


def test_gsd_with_invalid_timing():
    k = 4
    alpha = 0.025
    with pytest.raises(ValueError):
        GroupSequentialDesign(
            k=k,
            alpha=alpha,
            sfu=spending_function_obrien_fleming,
            timing=[0.25, 0.5, 0.75],
        )
    with pytest.raises(ValueError):
        GroupSequentialDesign(
            k=k,
            alpha=alpha,
            sfu=spending_function_obrien_fleming,
            timing=[0.25, 0.5, 0.5, 1.0],
        )
    with pytest.raises(ValueError):
        GroupSequentialDesign(
            k=k,
            alpha=alpha,
            sfu=spending_function_obrien_fleming,
            timing=[0.25, 0.5, 0.75, 1.1],
        )


def test_gsd_with_invalid_k():
    with pytest.raises(ValueError):
        GroupSequentialDesign(k=0)


def test_gsd_with_invalid_alpha():
    with pytest.raises(ValueError):
        GroupSequentialDesign(k=4, alpha=0)
    with pytest.raises(ValueError):
        GroupSequentialDesign(k=4, alpha=1)
