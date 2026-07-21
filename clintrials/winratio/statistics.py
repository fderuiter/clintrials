from __future__ import annotations
"""Statistical helpers for win-ratio simulations.

Random Seed Strategy: {statistics_seed_strategy}
"""


import numpy as np

from clintrials.core.stats import log_scale_p_value, log_scale_wald_interval


def calculate_confidence_intervals(wr: float, wins: int, losses: int):  # type: ignore
    """Calculate the 95% confidence intervals for the win ratio.

    Args:
        wr (float): The win ratio.
        wins (int): The number of wins.
        losses (int): The number of losses.

    Returns:
        tuple[float, float]: A tuple containing the lower and upper bounds of
            the confidence interval.
    """
    if wins == 0 or losses == 0:
        return (0, 0)
    variance = 1 / wins + 1 / losses
    standard_error = np.sqrt(variance)
    return log_scale_wald_interval(wr, standard_error)


def calculate_p_value(wr: float, wins: int, losses: int) -> float:
    """Calculate the p-value for the observed win ratio.

    Args:
        wr (float): The win ratio.
        wins (int): The number of wins.
        losses (int): The number of losses.

    Returns:
        float: The p-value.
    """
    if wins == 0 or losses == 0:
        return 1.0
    variance = 1 / wins + 1 / losses
    standard_error = np.sqrt(variance)
    return log_scale_p_value(wr, standard_error)


def calculate_win_ratio(wins: int, losses: int) -> float:
    """Calculate the win ratio.

    Args:
        wins (int): The number of wins.
        losses (int): The number of losses.

    Returns:
        float: The win ratio, or infinity if there are no losses.
    """
    if losses == 0:
        return float("inf")
    return wins / losses


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
