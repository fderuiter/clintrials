"""Statistical helpers for win-ratio simulations."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def calculate_confidence_intervals(wr: float, wins: int, losses: int):
    """
    Calculate the 95% confidence intervals for the win ratio.

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
    z_score = norm.ppf(0.975)
    log_wr = np.log(wr)
    lower_bound_log = log_wr - z_score * standard_error
    upper_bound_log = log_wr + z_score * standard_error
    lower_bound = np.exp(lower_bound_log)
    upper_bound = np.exp(upper_bound_log)
    return (lower_bound, upper_bound)


def calculate_p_value(wr: float, wins: int, losses: int) -> float:
    """
    Calculate the p-value for the observed win ratio.

    Args:
        wr (float): The win ratio.
        wins (int): The number of wins.
        losses (int): The number of losses.

    Returns:
        float: The p-value.
    """
    if wins == 0 or losses == 0:
        return 1.0
    observed_z = (np.log(wr)) / np.sqrt((1 / wins) + (1 / losses))
    return 2 * norm.sf(abs(observed_z))


def calculate_win_ratio(wins: int, losses: int) -> float:
    """
    Calculate the win ratio.

    Args:
        wins (int): The number of wins.
        losses (int): The number of losses.

    Returns:
        float: The win ratio, or infinity if there are no losses.
    """
    if losses == 0:
        return float("inf")
    return wins / losses
