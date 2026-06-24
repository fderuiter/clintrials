import numpy as np
import pytest

from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU


def test_watu_mc_config():
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

    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    # Test defaults
    trial = WATU(
        skeletons, tox_prior, tox_target, tox_cutoff, eff_cutoff, metric, 1, 64
    )
    assert trial.mc_sample_size == 10**5
    assert trial.mc_samples_stage1 == 10**5
    assert trial.mc_samples_stage2 == 10**5

    # Test initialization with custom values
    trial2 = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        1,
        64,
        mc_sample_size=2000,
        mc_samples_stage1=3000,
        mc_samples_stage2=4000,
    )
    assert trial2.mc_sample_size == 2000
    assert trial2.mc_samples_stage1 == 3000
    assert trial2.mc_samples_stage2 == 4000

    # Test clamping
    trial3 = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        1,
        64,
        mc_sample_size=500,
        mc_samples_stage1=600,
        mc_samples_stage2=700,
    )
    assert trial3.mc_sample_size == 1000
    assert trial3.mc_samples_stage1 == 1000
    assert trial3.mc_samples_stage2 == 1000

    # Test propagation of defaults
    trial4 = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        1,
        64,
        mc_sample_size=5000,
    )
    assert trial4.mc_sample_size == 5000
    assert trial4.mc_samples_stage1 == 5000
    assert trial4.mc_samples_stage2 == 5000


def test_watu_update_kwargs(mocker):
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30
    skeletons = [[0.60, 0.50, 0.40, 0.30, 0.20, 0.10]]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_cutoff,
        eff_cutoff,
        metric,
        1,
        64,
        stage_one_size=5,
    )

    cases = [(1, 0, 1)]

    # Mock prob_tox_exceeds and prob_eff_exceeds to check n
    mock_crm_exceeds = mocker.patch.object(
        trial.crm, "prob_tox_exceeds", return_value=np.array([0.1])
    )
    mock_eff_exceeds = mocker.patch.object(
        trial, "prob_eff_exceeds", return_value=np.array([0.9])
    )

    # Stage 1
    trial.update(cases, mc_samples_stage1=2500)
    mock_crm_exceeds.assert_called_with(trial.tox_limit)

    # Stage 2
    trial.stage_one_size = 0  # Force stage 2
    trial.update(cases, mc_samples_stage2=3500)
    mock_crm_exceeds.assert_called_with(trial.tox_limit)

    # General n
    trial.update(cases, n=4500)
    mock_crm_exceeds.assert_called_with(trial.tox_limit)
