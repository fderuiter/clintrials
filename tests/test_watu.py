# SPDX-License-Identifier: MIT

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
        skeletons=skeletons,
        prior_tox_probs=tox_prior,
        tox_target=tox_target,
        tox_limit=tox_cutoff,
        eff_limit=eff_cutoff,
        metric=metric,
        first_dose=first_dose,
        max_size=trial_size,
        stage_one_size=stage1_size,
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
        skeletons=skeletons,
        prior_tox_probs=tox_prior,
        tox_target=tox_target,
        tox_limit=tox_cutoff,
        eff_limit=eff_cutoff,
        metric=metric,
        first_dose=first_dose,
        max_size=trial_size,
        stage_one_size=stage1_size,
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
        np.abs(
            trial.post_tox_probs
            - np.array(
                [0.12922712, 0.31187137, 0.41243826, 0.49060208, 0.55690928, 0.61558775]
            )
        )
        < 0.00001
    )
    assert np.all(
        np.abs(
            trial.post_eff_probs
            - np.array([0.3999842, 0.4935573, 0.5830683, 0.6697644, 0.5830683, 0.4935573])
        )
        < 0.00001
    )
    assert np.all(
        np.abs(
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


def test_watu_prob_eff_exceeds_backends():
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30
    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
    ]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(
        skeletons=skeletons,
        prior_tox_probs=tox_prior,
        tox_target=tox_target,
        tox_limit=tox_cutoff,
        eff_limit=eff_cutoff,
        metric=metric,
        first_dose=1,
        max_size=64,
        stage_one_size=16,
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
    trial.update(cases)

    cutoff = 0.15
    probs_analytic = trial.prob_eff_exceeds(cutoff, backend="analytic")
    probs_quad = trial.prob_eff_exceeds(cutoff, backend="quadrature")
    probs_mc = trial.prob_eff_exceeds(cutoff, backend="mc", n=100000)

    assert len(probs_analytic) == 6
    assert len(probs_quad) == 6
    assert len(probs_mc) == 6

    assert np.all(np.abs(probs_analytic - probs_mc) < 0.02)
    assert np.all(np.abs(probs_analytic - probs_quad) < 0.03)
    assert np.all(np.abs(probs_quad - probs_mc) < 0.03)


def test_watu_prob_eff_exceeds_boundaries():
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    skeletons = [
        [1.0, 0.60, 0.50, 0.40, 0.10, 0.0]
    ]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(
        skeletons=skeletons,
        prior_tox_probs=tox_prior,
        tox_target=0.30,
        tox_limit=0.33,
        eff_limit=0.05,
        metric=metric,
        first_dose=1,
        max_size=64,
    )
    cases = [(1, 0, 1), (2, 0, 0)]
    trial.update(cases)

    probs_high = trial.prob_eff_exceeds(1.0, backend="analytic")
    assert np.all(probs_high == 0.0)

    probs_high_quad = trial.prob_eff_exceeds(1.5, backend="quadrature")
    assert np.all(probs_high_quad == 0.0)

    probs_low = trial.prob_eff_exceeds(-0.5, backend="analytic")
    assert np.all(probs_low == 1.0)

    probs_low_mc = trial.prob_eff_exceeds(-0.1, backend="mc")
    assert np.all(probs_low_mc == 1.0)

    probs_normal = trial.prob_eff_exceeds(0.2, backend="analytic")
    assert probs_normal[0] == 1.0
    assert probs_normal[5] == 0.0

    probs_quad = trial.prob_eff_exceeds(0.2, backend="quadrature")
    assert probs_quad[0] == 1.0
    assert probs_quad[5] == 0.0

    probs_mc = trial.prob_eff_exceeds(0.2, backend="mc")
    assert probs_mc[0] == 1.0
    assert probs_mc[5] == 0.0

