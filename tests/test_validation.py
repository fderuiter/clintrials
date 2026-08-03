from typing import Any

import pytest

from clintrials.validation import (
    validate_bounds,
    validate_expected_length,
    validate_matching_lengths,
    validate_positive_integer,
    validate_probability,
    validate_version,
)


def test_validate_matching_lengths():
    # Should not raise
    validate_matching_lengths()
    validate_matching_lengths(a=[1, 2], b=[3, 4])
    validate_matching_lengths(a=[1], b=[2], c=[3])

    # Should raise
    with pytest.raises(ValueError, match="a and b should be same length"):
        validate_matching_lengths(a=[1, 2], b=[3])
    with pytest.raises(ValueError, match="x and z should be same length"):
        validate_matching_lengths(x=[1, 2], y=[3, 4], z=[5, 6, 7])


def test_validate_expected_length():
    # Should not raise
    validate_expected_length([1, 2, 3], 3, "test_arr")

    # Should raise
    with pytest.raises(ValueError, match="test_arr should have 3 items"):
        validate_expected_length([1, 2], 3, "test_arr")


def test_validate_bounds():
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


def test_validate_probability():
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


def test_validate_positive_integer():
    validate_positive_integer(1, "val")
    validate_positive_integer(100, "val")

    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer(0, "val")
    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer(-1, "val")
    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer(1.5, "val")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="val must be a positive integer"):
        validate_positive_integer("1", "val")  # type: ignore[arg-type]


def test_validate_version():
    # Valid PEP 440 versions should not raise
    validate_version("1.0.0", "version")
    validate_version("2.1.0b1", "version")
    validate_version("0.1.4", "version")
    validate_version("1.2.3.post1", "version")
    validate_version("3.0.0a2", "version")
    validate_version("1.0.0-beta", "version")

    # Invalid PEP 440 versions should raise ValueError with correct message
    invalid_versions = ["latest", "stable", "", "abc"]
    for val in invalid_versions:
        with pytest.raises(ValueError, match="version must be a valid PEP 440 version string"):
            validate_version(val, "version")

    # Non-string values should raise ValueError with correct message
    non_strings: list[Any] = [None, 123, 1.0, [], {}, True]
    for non_str in non_strings:
        with pytest.raises(ValueError, match="version must be a valid PEP 440 version string"):
            validate_version(non_str, "version")


def test_new_schema_and_math_guardrails():
    import numpy as np

    from clintrials.core.stats import (
        ProbabilityDensitySample,
        log_scale_p_value,
        log_scale_wald_interval,
    )
    from clintrials.dosefinding.efftox import LpNormCurve
    from clintrials.dosefinding.wagestait import WagesTait
    from clintrials.dosefinding.watu import WATU
    from clintrials.phase3.gsd import GroupSequentialDesign
    from clintrials.winratio.compare import compare_subjects
    from clintrials.winratio.simulate import simulate_comparisons
    from clintrials.winratio.statistics import (
        calculate_confidence_intervals,
        calculate_p_value,
        calculate_win_ratio,
    )

    skeletons = [[0.1, 0.2, 0.3]]
    prior_tox_probs = [0.1, 0.2, 0.3]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    # 1. WagesTait validation: tox_target > tox_limit
    with pytest.raises(ValueError, match="tox_target must be <= tox_limit"):
        WagesTait(skeletons, prior_tox_probs, tox_target=0.4, tox_limit=0.3, eff_limit=0.1, first_dose=1, max_size=30, randomisation_stage_size=10)

    # WagesTait validation: negative max_size (boundary check via schema)
    with pytest.raises(ValueError, match="max_size must be a positive integer"):
        WagesTait(skeletons, prior_tox_probs, tox_target=0.2, tox_limit=0.3, eff_limit=0.1, first_dose=1, max_size=-5, randomisation_stage_size=10)

    # 2. WATU validation: tox_target > tox_limit
    with pytest.raises(ValueError, match="tox_target must be <= tox_limit"):
        WATU(skeletons, prior_tox_probs, tox_target=0.4, tox_limit=0.3, eff_limit=0.1, metric=metric, first_dose=1, max_size=30)

    # WATU validation: invalid probability in skeletons
    bad_skeletons = [[-0.1, 0.2, 0.3]]
    with pytest.raises(ValueError, match="skeletons must be between 0.0 and 1.0"):
        WATU(bad_skeletons, prior_tox_probs, tox_target=0.2, tox_limit=0.3, eff_limit=0.1, metric=metric, first_dose=1, max_size=30)

    # 3. GroupSequentialDesign validation: alpha <= 0
    with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
        GroupSequentialDesign(k=3, alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
        GroupSequentialDesign(k=3, alpha=1.0)

    # 4. Subject comparisons sequence / type validation
    # compare_subjects
    with pytest.raises(TypeError, match="Subject outcomes must be sequence types"):
        compare_subjects(1, [2, 3])  # type: ignore

    with pytest.raises(ValueError, match="subject1 and subject2 should be same length"):
        compare_subjects([1, 2], [1, 2, 3])

    # simulate_comparisons
    with pytest.raises(TypeError, match="Groups must be sequence or array types"):
        simulate_comparisons(1, [[2, 3]])

    with pytest.raises(ValueError, match="treatment_group components and control_group components should be same length"):
        simulate_comparisons([[1, 2]], [[1, 2, 3]])

    # 5. Statistical/math checks
    # Log scale ratio validation
    with pytest.raises(ValueError, match="ratio must be > 0"):
        log_scale_wald_interval(ratio=0.0, standard_error=0.1)

    with pytest.raises(ValueError, match="standard_error must be > 0"):
        log_scale_wald_interval(ratio=1.5, standard_error=0.0)

    with pytest.raises(ValueError, match="ratio must be > 0"):
        log_scale_p_value(ratio=-0.5, standard_error=0.1)

    with pytest.raises(ValueError, match="standard_error must be > 0"):
        log_scale_p_value(ratio=1.5, standard_error=-0.1)

    # ProbabilityDensitySample scale division-by-zero validation
    with pytest.raises(ValueError, match="scale must be > 0"):
        ProbabilityDensitySample(np.array([1, 2]), lambda x: np.array([0.0, 0.0]))

    # Negative wins/losses validation in winratio statistics
    with pytest.raises(ValueError, match="wins must be >= 0"):
        calculate_confidence_intervals(1.5, -1, 5)

    with pytest.raises(ValueError, match="losses must be >= 0"):
        calculate_p_value(1.5, 3, -2)

    with pytest.raises(ValueError, match="losses must be >= 0"):
        calculate_win_ratio(3, -2)

    # Division-by-zero validation (wins or losses is 0)
    with pytest.raises(ValueError, match="wins must be > 0"):
        calculate_confidence_intervals(1.5, 0, 5)
    with pytest.raises(ValueError, match="losses must be > 0"):
        calculate_confidence_intervals(1.5, 5, 0)
    with pytest.raises(ValueError, match="losses must be > 0"):
        calculate_win_ratio(5, 0)

    # Log of negative or zero ratio validation
    with pytest.raises(ValueError, match="wr must be > 0"):
        calculate_confidence_intervals(0.0, 5, 5)
    with pytest.raises(ValueError, match="wr must be > 0"):
        calculate_confidence_intervals(-1.5, 5, 5)
    with pytest.raises(ValueError, match="wr must be > 0"):
        calculate_p_value(0.0, 5, 5)
    with pytest.raises(ValueError, match="wr must be > 0"):
        calculate_p_value(-1.5, 5, 5)

    # Unmatched sequence shapes or invalid input types for compare_subjects
    with pytest.raises(ValueError, match="Subject outcomes must be 1D sequences."):
        compare_subjects(np.array([[1, 2]]), np.array([1, 2]))
    with pytest.raises(ValueError, match="subject1 and subject2 should be same length"):
        compare_subjects(np.array([1, 2]), np.array([1, 2, 3]))
