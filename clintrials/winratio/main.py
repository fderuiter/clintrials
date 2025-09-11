"""Command-line entry point for win-ratio power simulations."""

from __future__ import annotations

import argparse

from .data_generation import generate_data
from .simulate import simulate_comparisons
from .statistics import (
    calculate_confidence_intervals,
    calculate_p_value,
    calculate_win_ratio,
)


def run_simulation(
    num_subjects_A: int,
    num_subjects_B: int,
    num_simulations: int,
    p_y1_A: float,
    p_y1_B: float,
    p_y2_A: float,
    p_y2_B: float,
    p_y3_A: float,
    p_y3_B: float,
    significance_level: float = 0.05,
):
    """
    Run a Monte Carlo simulation to estimate win-ratio power.

    Args:
        num_subjects_A (int): The number of subjects in Group A.
        num_subjects_B (int): The number of subjects in Group B.
        num_simulations (int): The number of simulations to run.
        p_y1_A (float): The probability of outcome y1=1 for Group A.
        p_y1_B (float): The probability of outcome y1=1 for Group B.
        p_y2_A (float): The probability of outcome y2=1 for Group A.
        p_y2_B (float): The probability of outcome y2=1 for Group B.
        p_y3_A (float): The probability of outcome y3=1 for Group A.
        p_y3_B (float): The probability of outcome y3=1 for Group B.
        significance_level (float, optional): The significance level for the
            p-value. Defaults to 0.05.

    Returns:
        tuple[float, tuple[float, float]]: A tuple containing the estimated
            power and the average confidence interval.
    """
    successes = 0
    all_cis = []
    for _ in range(num_simulations):
        treatment_group, control_group = generate_data(
            num_subjects_A,
            num_subjects_B,
            p_y1_A,
            p_y1_B,
            p_y2_A,
            p_y2_B,
            p_y3_A,
            p_y3_B,
        )
        results = simulate_comparisons(treatment_group, control_group)
        wr = calculate_win_ratio(results["wins"], results["losses"])
        if wr == float("inf"):
            if results["wins"] > 0:
                successes += 1
            continue
        ci = calculate_confidence_intervals(wr, results["wins"], results["losses"])
        p_value = calculate_p_value(wr, results["wins"], results["losses"])
        all_cis.append(ci)
        if p_value < significance_level:
            successes += 1
    power = successes / num_simulations
    if all_cis:
        average_ci = (
            sum(ci[0] for ci in all_cis) / len(all_cis),
            sum(ci[1] for ci in all_cis) / len(all_cis),
        )
    else:
        average_ci = (0, 0)
    return power, average_ci


def main() -> None:
    """Parse command-line arguments and run the simulation."""
    parser = argparse.ArgumentParser(
        description="Run a win-ratio simulation to calculate statistical power."
    )
    parser.add_argument(
        "--num_subjects_A", type=int, default=100, help="Number of subjects in Group A."
    )
    parser.add_argument(
        "--num_subjects_B", type=int, default=50, help="Number of subjects in Group B."
    )
    parser.add_argument(
        "--num_simulations", type=int, default=10000, help="Number of simulations."
    )
    parser.add_argument(
        "--p_y1_A", type=float, default=0.50, help="Probability of y1=1 for Group A."
    )
    parser.add_argument(
        "--p_y1_B", type=float, default=0.50, help="Probability of y1=1 for Group B."
    )
    parser.add_argument(
        "--p_y2_A", type=float, default=0.75, help="Probability of y2=1 for Group A."
    )
    parser.add_argument(
        "--p_y2_B", type=float, default=0.25, help="Probability of y2=1 for Group B."
    )
    parser.add_argument(
        "--p_y3_A", type=float, default=0.43, help="Probability of y3=1 for Group A."
    )
    parser.add_argument(
        "--p_y3_B", type=float, default=0.27, help="Probability of y3=1 for Group B."
    )
    parser.add_argument(
        "--significance", type=float, default=0.05, help="Significance level."
    )
    args = parser.parse_args()
    power, average_ci = run_simulation(
        args.num_subjects_A,
        args.num_subjects_B,
        args.num_simulations,
        args.p_y1_A,
        args.p_y1_B,
        args.p_y2_A,
        args.p_y2_B,
        args.p_y3_A,
        args.p_y3_B,
        args.significance,
    )
    print(f"Power of the test: {power:.4f}")  # noqa: T201
    print(  # noqa: T201
        "Average confidence interval: " f"({average_ci[0]:.4f}, {average_ci[1]:.4f})"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
