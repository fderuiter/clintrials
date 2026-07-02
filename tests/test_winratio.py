import numpy as np

from clintrials.winratio import (
    calculate_confidence_intervals,
    calculate_p_value,
    calculate_win_ratio,
    compare_subjects,
    generate_data,
    simulate_comparisons,
)


def test_compare_subjects():
    assert compare_subjects([1, 0, 0], [0, 0, 0]) == "win"
    assert compare_subjects([0, 0, 0], [1, 0, 0]) == "loss"
    assert compare_subjects([1, 0, 1], [1, 0, 1]) == "tie"


def test_generate_data_shapes():
    a, b = generate_data(5, 4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert a.shape == (5, 3)
    assert b.shape == (4, 3)


def test_simulate_comparisons_counts():
    treatment = np.array([[1, 0, 0]])
    control = np.array([[0, 0, 0], [1, 0, 0]])
    results = simulate_comparisons(treatment, control)
    assert results == {"wins": 1, "losses": 0, "ties": 1}


def test_statistics_helpers():
    wr = calculate_win_ratio(20, 10)
    assert wr == 2
    ci = calculate_confidence_intervals(wr, 20, 10)
    assert len(ci) == 2
    p_val = calculate_p_value(wr, 20, 10)
    assert 0 <= p_val <= 1


from unittest.mock import patch
from clintrials.winratio.main import WinRatioTrial, main


def test_winratio_trial():
    trial = WinRatioTrial(
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
    assert trial.has_more() is True
    trial.update()
    assert trial.has_more() is False
    report = trial.report()
    assert "success" in report
    assert "ci" in report
    trial.reset()
    assert trial.success is False
    assert trial.has_more() is True


@patch(
    "sys.argv",
    [
        "main",
        "--num_subjects_A",
        "10",
        "--num_subjects_B",
        "10",
        "--num_simulations",
        "5",
    ],
)
def test_main_cli():
    main()
