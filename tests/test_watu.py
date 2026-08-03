__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

""" Tests of the clintrials.dosefindings.wagestait module. """

import numpy as np

from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU


def setup_func():
    pass


def teardown_func():
    pass


def test_watu_1():

    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30

    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        [0.30, 0.40, 0.50, 0.60, 0.50, 0.40],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.60, 0.60],
        [0.40, 0.50, 0.60, 0.60, 0.60, 0.60],
        [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    ]

    first_dose = 1
    trial_size = 64
    stage1_size = 16

    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        first_dose,
        trial_size,
        stage1_size,
    )

    cases = [
        (1, 1, 0),
        (1, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (2, 0, 0),
        (2, 0, 1),
        (3, 1, 1),
        (3, 0, 1),
    ]

    next_dose = trial.update(cases)
    assert next_dose == 2

    assert np.all(
        np.abs(
            trial.post_tox_probs
            - np.array(
                [0.13749075, 0.31266169, 0.40958313, 0.48560574, 0.55065048, 0.60866504]
            )
        )
        < 0.001
    )  # First one varies a bit more
    assert np.all(
        np.abs(
            trial.post_eff_probs
            - np.array(
                [0.2479070, 0.3639813, 0.4615474, 0.5497718, 0.6321674, 0.7105235]
            )
        )
        < 0.00001
    )
    assert np.all(
        np.abs(
            trial.w
            - np.array(
                [
                    0.01347890,
                    0.03951504,
                    0.12006585,
                    0.11798287,
                    0.11764227,
                    0.12346595,
                    0.11764227,
                    0.11798287,
                    0.12006585,
                    0.07073296,
                    0.04142517,
                ]
            )
        )
        < 0.00001
    )
    assert trial.most_likely_model_index == 5
    # Admissible doses remain
    assert trial.admissable_set() == [1, 2, 3, 4]
    # Probabilities and weights should have correct lengths
    assert len(trial.post_tox_probs) == 6
    assert len(trial.post_eff_probs) == 6
    assert len(trial.w) == 11
    # Stage 1 utility is still empty
    assert trial.utility == []


def test_watu_2():

    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30

    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        [0.30, 0.40, 0.50, 0.60, 0.50, 0.40],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.60, 0.60],
        [0.40, 0.50, 0.60, 0.60, 0.60, 0.60],
        [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    ]

    first_dose = 1
    trial_size = 64
    stage1_size = 16

    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        first_dose,
        trial_size,
        stage1_size,
    )

    cases = [
        (1, 1, 0),
        (1, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (2, 0, 0),
        (2, 0, 1),
        (3, 1, 1),
        (3, 0, 1),
        (3, 1, 1),
        (2, 0, 0),
        (2, 0, 0),
        (2, 1, 1),
        (3, 0, 1),
        (3, 0, 0),
        (3, 1, 1),
        (4, 1, 1),
        (4, 0, 1),
        (4, 0, 1),
    ]

    next_dose = trial.update(cases)
    assert next_dose == 1
    assert np.all(
        trial.post_tox_probs
        - np.array(
            [0.12922712, 0.31187137, 0.41243826, 0.49060208, 0.55690928, 0.61558775]
        )
        < 0.00001
    )
    assert np.all(
        trial.post_eff_probs
        - np.array([0.3999842, 0.4935573, 0.5830683, 0.6697644, 0.5830683, 0.4935573])
        < 0.00001
    )
    assert np.all(
        trial.w
        - np.array(
            [
                0.00165319,
                0.00650976,
                0.06932715,
                0.15695883,
                0.14129752,
                0.14465125,
                0.14129752,
                0.15695883,
                0.11767193,
                0.04176601,
                0.02190798,
            ]
        )
        < 0.00001
    )
    assert trial.most_likely_model_index == 3
    # Admissible doses remain
    assert trial.admissable_set() == [1, 2, 3, 4]
    # Returned arrays should have expected lengths
    assert len(trial.post_tox_probs) == 6
    assert len(trial.post_eff_probs) == 6
    assert len(trial.w) == 11

    # Utility is now a non-empty array in stage 2
    assert np.all(
        np.abs(
            trial.utility
            - np.array(
                [
                    0.18320154,
                    -0.11034328,
                    -0.26984169,
                    -0.39399425,
                    -0.61068672,
                    -0.81190408,
                ]
            )
        )
        < 0.00001
    )
