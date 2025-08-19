import numpy as np

from clintrials.winratio import (
    calculate_confidence_intervals,
    calculate_p_value,
    calculate_win_ratio,
    compare_subjects,
    generate_data,
    run_simulation,
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


def test_run_simulation_executes():
    np.random.seed(0)
    power, ci = run_simulation(5, 5, 10, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4)
    assert 0 <= power <= 1
    assert len(ci) == 2
