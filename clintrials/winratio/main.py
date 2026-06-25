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

from collections import OrderedDict
from clintrials.core.protocol import Protocol
from clintrials.core.schema import WinRatioSchema
from clintrials.core.simulation import run_sims

class WinRatioTrial(Protocol):
    """Win-Ratio simulation wrapped as a Protocol."""
    
    def __init__(self, **kwargs):
        self.config = WinRatioSchema(**kwargs)
        self.power = 0.0
        self.average_ci = (0.0, 0.0)
        self._completed = False

    def reset(self):
        """Reset the simulation state."""
        self.power = 0.0
        self.average_ci = (0.0, 0.0)
        self._completed = False

    def update(self, *args, **kwargs):
        """Run the simulation to update the state."""
        self.power, self.average_ci = run_simulation(
            num_subjects_A=self.config.num_subjects_A,
            num_subjects_B=self.config.num_subjects_B,
            num_simulations=self.config.num_simulations,
            p_y1_A=self.config.p_y1_A,
            p_y1_B=self.config.p_y1_B,
            p_y2_A=self.config.p_y2_A,
            p_y2_B=self.config.p_y2_B,
            p_y3_A=self.config.p_y3_A,
            p_y3_B=self.config.p_y3_B,
            significance_level=self.config.significance_level,
        )
        self._completed = True

    def has_more(self):
        """Check if simulation is completed."""
        return not self._completed

    def report(self):
        """Report the trial results."""
        return OrderedDict([
            ("power", self.power),
            ("average_ci", self.average_ci)
        ])


def _single_iteration(
    num_subjects_A: int,
    num_subjects_B: int,
    p_y1_A: float,
    p_y1_B: float,
    p_y2_A: float,
    p_y2_B: float,
    p_y3_A: float,
    p_y3_B: float,
    significance_level: float,
):
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
        return results["wins"] > 0, None
    ci = calculate_confidence_intervals(wr, results["wins"], results["losses"])
    p_value = calculate_p_value(wr, results["wins"], results["losses"])
    return p_value < significance_level, ci


def _winratio_agg_func(current_sims, new_batch_sims):
    if current_sims is None:
        successes = 0
        total_sims = 0
        sum_ci0 = 0.0
        sum_ci1 = 0.0
        ci_count = 0
    else:
        successes, total_sims, sum_ci0, sum_ci1, ci_count = current_sims

    for success, ci in new_batch_sims:
        if success:
            successes += 1
        total_sims += 1
        if ci is not None:
            sum_ci0 += ci[0]
            sum_ci1 += ci[1]
            ci_count += 1

    return successes, total_sims, sum_ci0, sum_ci1, ci_count


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
    state = run_sims(
        sim_func=_single_iteration,
        n1=1,
        n2=num_simulations,
        agg_func=_winratio_agg_func,
        num_subjects_A=num_subjects_A,
        num_subjects_B=num_subjects_B,
        p_y1_A=p_y1_A,
        p_y1_B=p_y1_B,
        p_y2_A=p_y2_A,
        p_y2_B=p_y2_B,
        p_y3_A=p_y3_A,
        p_y3_B=p_y3_B,
        significance_level=significance_level,
    )
    if state is None:
        return 0.0, (0.0, 0.0)
    successes, total_sims, sum_ci0, sum_ci1, ci_count = state
    power = successes / total_sims if total_sims > 0 else 0.0
    if ci_count > 0:
        average_ci = (sum_ci0 / ci_count, sum_ci1 / ci_count)
    else:
        average_ci = (0, 0)
    return power, average_ci


def main() -> None:
    """Parse command-line arguments and run the simulation."""
    parser = argparse.ArgumentParser(
        description="Run a win-ratio simulation to calculate statistical power."
    )
    for name, field in WinRatioSchema.model_fields.items():
        arg_name = f"--{name}"
        if name == "significance_level":
            arg_name = "--significance"
        
        arg_type = int if "PositiveInt" in str(field.annotation) else float
        parser.add_argument(
            arg_name,
            type=arg_type,
            default=field.default,
            help=field.description,
        )

    args = parser.parse_args()
    
    # Map back significance to significance_level if passed
    kwargs = vars(args)
    if "significance" in kwargs:
        kwargs["significance_level"] = kwargs.pop("significance")

    trial = WinRatioTrial(**kwargs)
    trial.update()
    power = trial.power
    average_ci = trial.average_ci
    
    print(f"Power of the test: {power:.4f}")  # noqa: T201
    print(  # noqa: T201
        "Average confidence interval: " f"({average_ci[0]:.4f}, {average_ci[1]:.4f})"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
