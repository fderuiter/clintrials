import numpy as np
from scipy.stats import norm
from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU

def test_watu_must_try_lowest_dose():
    # Simple skeletons to ensure admissibility
    tox_prior = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    tox_cutoff = 0.4
    eff_cutoff = 0.1
    tox_target = 0.2

    skeletons = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ]

    metric = LpNormCurve(0.1, 0.4, 0.2, 0.2)

    # Scenario 1: must_try_lowest_dose=True, first_dose=3
    # Trial should start at dose 1
    trial_t = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        first_dose=3,
        max_size=30,
        stage_one_size=10, # Ensure stage 1
        must_try_lowest_dose=True
    )
    assert trial_t.next_dose() == 1

    # Scenario 2: must_try_lowest_dose=False (default), first_dose=3
    # Trial should start at dose 3
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
        must_try_lowest_dose=False
    )
    assert trial_f.next_dose() == 3

    # Scenario 3: After the first cohort, must_try_lowest_dose should not force dose 1
    # Add a safe outcome at dose 1. We expect it to escalate since it's very safe.
    trial_t.update([(1, 0, 1)])
    # It should recommend something else than 1 if 1 is not the most attractive.
    # With tox target 0.2 and dose 1 having tox prior 0.01, it should escalate.
    assert trial_t.next_dose() > 1

def test_watu_reset_honors_flag():
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    skeletons = [[0.1]*6]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    trial = WATU(
        skeletons,
        tox_prior,
        0.30, 0.33, 0.05,
        metric,
        first_dose=3,
        max_size=30,
        must_try_lowest_dose=True
    )
    assert trial.next_dose() == 1
    trial.update([(1, 0, 0)])
    trial.reset()
    assert trial.next_dose() == 1
