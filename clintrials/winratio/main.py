"""Command-line entry point for win-ratio power simulations.

Random Seed Strategy: {main_seed_strategy}
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

from clintrials.core.protocol import Protocol
from clintrials.core.schema import WinRatioSchema

from .data_generation import generate_data
from .simulate import simulate_comparisons
from .statistics import (
    calculate_confidence_intervals,
    calculate_p_value,
    calculate_win_ratio,
)


class WinRatioTrial(Protocol):
    """Win-Ratio simulation wrapped as a Protocol."""

    def __init__(self, **kwargs):
        super().__init__()
        self.config = WinRatioSchema(**kwargs)
        self.success = False
        self.ci = None
        self._completed = False

    def reset(self):
        """Reset the simulation state."""
        self.success = False
        self.ci = None
        self._completed = False

    def update(self, *args, **kwargs):
        """Run a single trial simulation to update the state."""
        self.success, self.ci = _single_iteration(
            num_subjects_A=self.config.num_subjects_A,
            num_subjects_B=self.config.num_subjects_B,
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
        """Check if trial is completed."""
        return not self._completed

    def report(self):
        """Report the trial results."""
        return OrderedDict([("success", self.success), ("ci", self.ci)])


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


from clintrials.utils import Memoize


@Memoize
def run_winratio_simulations(**kwargs):
    """Run win-ratio simulation using the core runner and calculate summary metrics.

    Returns a dictionary with 'power', 'average_ci', and the raw 'results'.
    """
    trial = WinRatioTrial(**kwargs)

    # Run bulk simulations via unified runner
    num_simulations = getattr(trial.config, 'num_simulations', 1)
    results = trial.run(n_sims=num_simulations, method="iterative")

    # Extract list of result dicts, depending on if SimulationResult is iterable or has a property
    # SimulationResult is iterable
    results_list = list(results)

    successes = sum(1 for r in results_list if r.get("success", False))
    total_sims = len(results_list)

    sum_ci0 = 0.0
    sum_ci1 = 0.0
    ci_count = 0
    for r in results_list:
        if r.get("ci") is not None:
            sum_ci0 += r["ci"][0]
            sum_ci1 += r["ci"][1]
            ci_count += 1

    power = successes / total_sims if total_sims > 0 else 0.0
    if ci_count > 0:
        average_ci = (sum_ci0 / ci_count, sum_ci1 / ci_count)
    else:
        average_ci = (0, 0)

    return {
        "power": power,
        "average_ci": average_ci,
        "results": results_list
    }

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

    summary = run_winratio_simulations(**kwargs)
    power = summary["power"]
    average_ci = summary["average_ci"]

    print(f"Power of the test: {power:.4f}")  # noqa: T201
    print(  # noqa: T201
        "Average confidence interval: " f"({average_ci[0]:.4f}, {average_ci[1]:.4f})"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
