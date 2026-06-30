import numpy as np
from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU


def test_watu_must_try_lowest_dose():
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30
    skeletons = [[0.60, 0.50, 0.40, 0.30, 0.20, 0.10]]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    # Scenario 1: must_try_lowest_dose=True, first_dose=3
    trial_t = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        first_dose=3,
        max_size=30,
        stage_one_size=10,
        must_try_lowest_dose=True,
    )
    assert trial_t.next_dose() == 1

    # Scenario 2: must_try_lowest_dose=False
    trial_f = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        first_dose=3,
        max_size=30,
        stage_one_size=10,
        must_try_lowest_dose=False,
    )
    assert trial_f.next_dose() == 3


def test_watu_reset_honors_flag():
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    skeletons = [[0.1] * 6]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(
        skeletons,
        tox_prior,
        0.30,
        0.33,
        0.05,
        metric,
        first_dose=3,
        max_size=30,
        must_try_lowest_dose=True,
    )
    assert trial.next_dose() == 1
    trial.reset()
    assert trial.next_dose() == 1
