import pytest

from clintrials.validation import (
    validate_bounds,
    validate_expected_length,
    validate_matching_lengths,
    validate_positive_integer,
    validate_probability,
)


def test_validate_matching_lengths():  # type: ignore
    # Should not raise
    validate_matching_lengths()  # type: ignore
    validate_matching_lengths(a=[1, 2], b=[3, 4])  # type: ignore
    validate_matching_lengths(a=[1], b=[2], c=[3])  # type: ignore

    # Should raise
    with pytest.raises(ValueError, match="a and b should be same length"):
        validate_matching_lengths(a=[1, 2], b=[3])  # type: ignore
    with pytest.raises(ValueError, match="x and z should be same length"):
        validate_matching_lengths(x=[1, 2], y=[3, 4], z=[5, 6, 7])  # type: ignore


def test_validate_expected_length():  # type: ignore
    # Should not raise
    validate_expected_length([1, 2, 3], 3, "test_arr")

    # Should raise
    with pytest.raises(ValueError, match="test_arr should have 3 items"):
        validate_expected_length([1, 2], 3, "test_arr")


def test_validate_bounds():  # type: ignore
    # Inclusive
    validate_bounds(5, 0, 10, "val", exclusive=False)
    validate_bounds(0, 0, 10, "val", exclusive=False)
    validate_bounds(10, 0, 10, "val", exclusive=False)

    with pytest.raises(ValueError, match="val must be >= 0"):
        validate_bounds(-1, 0, 10, "val", exclusive=False)
    with pytest.raises(ValueError, match="val must be <= 10"):
        validate_bounds(11, 0, 10, "val", exclusive=False)

    # Exclusive
    validate_bounds(5, 0, 10, "val", exclusive=True)

    with pytest.raises(ValueError, match="val must be > 0"):
        validate_bounds(0, 0, 10, "val", exclusive=True)
    with pytest.raises(ValueError, match="val must be < 10"):
        validate_bounds(10, 0, 10, "val", exclusive=True)


def test_validate_probability():  # type: ignore
    # Inclusive
    validate_probability(0.5, "prob")
    validate_probability(0, "prob")
    validate_probability(1, "prob")

    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0"):
        validate_probability(-0.1, "prob")
    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0"):
        validate_probability(1.1, "prob")

    # Exclusive
    validate_probability(0.5, "prob", exclusive=True)

    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0"):
        validate_probability(0, "prob", exclusive=True)
    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0"):
        validate_probability(1, "prob", exclusive=True)


def test_validate_positive_integer():  # type: ignore
    validate_positive_integer(1, "val")
    validate_positive_integer(100, "val")

    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer(0, "val")
    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer(-1, "val")
    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer(1.5, "val")
    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer("1", "val")
