import numpy as np
from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU


def test_watu_stage_aware_mc_config(mocker):
    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30
    skeletons = [[0.60, 0.50, 0.40, 0.30, 0.20, 0.10]]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    # Initialise trial with custom stage defaults
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
        mc_samples_stage1=2000,
        mc_samples_stage2=4000,
    )

    # Mock the CRM and Efficacy resolution to see what 'n' is passed
    mock_crm_exceeds = mocker.patch.object(
        trial.crm, "prob_tox_exceeds", return_value=np.array([0.1] * 6)
    )

    # We must ensure model_theta_var and model_theta_hat are properly mocked to avoid analytic failure
    mocker.patch.object(trial, "model_theta_var", return_value=1.0)
    mocker.patch.object(trial, "model_theta_hat", return_value=0.0)
    trial.estimate_var = True

    # 1. Stage 1 logic (trial.size() < 5)
    trial.prob_acc_tox()
    mock_crm_exceeds.assert_called_with(trial.tox_limit, n=2000)

    # Priority check: mc_samples_stage2 should be ignored in Stage 1
    trial.prob_acc_tox(mc_samples_stage1=3000, mc_samples_stage2=9000)
    mock_crm_exceeds.assert_called_with(trial.tox_limit, n=3000)

    # CRM Kwargs preservation
    trial.prob_acc_tox(epsabs=1e-5)
    mock_crm_exceeds.assert_called_with(trial.tox_limit, n=2000, epsabs=1e-5)

    # 2. Stage 2 logic (trial.size() >= 5)
    trial._doses = [1, 2, 3, 4, 5]
    trial._toxicities = [0, 0, 0, 0, 0]
    trial._efficacies = [1, 1, 1, 1, 1]

    trial.prob_acc_tox()
    mock_crm_exceeds.assert_called_with(trial.tox_limit, n=4000)

    # Priority check: mc_samples_stage1 should be ignored in Stage 2
    trial.prob_acc_tox(mc_samples_stage1=9000, mc_samples_stage2=5000)
    mock_crm_exceeds.assert_called_with(trial.tox_limit, n=5000)

    # 3. Update call integration
    trial.update([(1, 0, 1)], mc_samples_stage2=8000)
    mock_crm_exceeds.assert_called_with(trial.tox_limit, n=8000)
