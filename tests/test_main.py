from clintrials.winratio.main import WinRatioTrial


def test_winratio_trial_run_reproducibility():
    trial1 = WinRatioTrial(
        num_subjects_A=10,
        num_subjects_B=10,
        num_simulations=5,
        p_y1_A=0.6,
        p_y1_B=0.4,
        p_y2_A=0.6,
        p_y2_B=0.4,
        p_y3_A=0.6,
        p_y3_B=0.4,
        significance_level=0.05,
    )
    results1 = list(trial1.run(n_sims=5, method="iterative", seed=100))

    trial2 = WinRatioTrial(
        num_subjects_A=10,
        num_subjects_B=10,
        num_simulations=5,
        p_y1_A=0.6,
        p_y1_B=0.4,
        p_y2_A=0.6,
        p_y2_B=0.4,
        p_y3_A=0.6,
        p_y3_B=0.4,
        significance_level=0.05,
    )
    results2 = list(trial2.run(n_sims=5, method="iterative", seed=100))

    trial3 = WinRatioTrial(
        num_subjects_A=10,
        num_subjects_B=10,
        num_simulations=5,
        p_y1_A=0.6,
        p_y1_B=0.4,
        p_y2_A=0.6,
        p_y2_B=0.4,
        p_y3_A=0.6,
        p_y3_B=0.4,
        significance_level=0.05,
    )
    results3 = list(trial3.run(n_sims=5, method="iterative", seed=200))

    assert results1 == results2
    assert results1 != results3
