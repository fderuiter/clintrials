"""
Module for Group Sequential Designs (GSDs).
"""

from typing import Callable, List

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm


def spending_function_pocock(t: float, alpha: float) -> float:
    """Pocock-like alpha spending function.

    This function implements the Pocock-like alpha spending function by Lan
    and DeMets.

    Args:
        t (float): The fraction of information accrued, between 0 and 1.
        alpha (float): The total significance level.

    Returns:
        float: The cumulative alpha spent at information fraction `t`.
    """
    return alpha * np.log(1 + (np.e - 1) * t)


def spending_function_obrien_fleming(t: float, alpha: float) -> float:
    """O'Brien-Fleming-like alpha spending function.

    This function implements the O'Brien-Fleming-like alpha spending function
    by Lan and DeMets.

    Args:
        t (float): The fraction of information accrued, between 0 and 1.
        alpha (float): The total significance level.

    Returns:
        float: The cumulative alpha spent at information fraction `t`.
    """
    if t == 0:
        return 0.0
    return 2 * (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(t)))


class GroupSequentialDesign:
    """A class to represent a group sequential design.

    This class calculates the efficacy boundaries for a group sequential trial
    given the number of looks, the significance level, and a spending function.
    """

    def __init__(
        self,
        k: int,
        alpha: float = 0.025,
        sfu: Callable[[float, float], float] = spending_function_obrien_fleming,
        timing: List[float] = None,
    ):
        """Initializes a GroupSequentialDesign object.

        Args:
            k (int): The number of analyses (looks) in the trial.
            alpha (float, optional): The overall one-sided significance level.
                Defaults to 0.025.
            sfu (Callable[[float, float], float], optional): The upper
                (efficacy) spending function. Defaults to
                `spending_function_obrien_fleming`.
            timing (List[float], optional): A list of information fractions for
                each look. If `None`, assumes equally spaced looks. Defaults to
                `None`.

        Raises:
            ValueError: If `alpha` is not between 0 and 1, `k` is not a
                positive integer, or `timing` is not a strictly increasing
                sequence of length `k` ending in 1.0.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if k < 1:
            raise ValueError("k must be a positive integer.")

        self.k = k
        self.alpha = alpha
        self.sfu = sfu
        self.timing = timing if timing is not None else np.linspace(1 / k, 1, k)

        if len(self.timing) != k:
            raise ValueError("Length of timing must be equal to k.")
        if any(self.timing[i] >= self.timing[i + 1] for i in range(k - 1)):
            raise ValueError("Timing must be strictly increasing.")
        if self.timing[-1] != 1.0:
            raise ValueError("The last element of timing must be 1.0.")

        self.efficacy_boundaries = self._compute_efficacy_boundaries()

    def _compute_efficacy_boundaries(self) -> List[float]:
        """Computes the efficacy boundaries for the design.

        This is done by finding the boundary u_i at each look i such that
        P(Z_1 < u_1, ..., Z_i < u_i) = 1 - alpha_i, where alpha_i is the
        cumulative alpha spent at look i.
        """
        boundaries = []
        for i in range(1, self.k + 1):
            target_alpha = self.sfu(self.timing[i - 1], self.alpha)

            target_cdf = 1 - target_alpha

            cov = np.identity(i)
            for row in range(i):
                for col in range(row + 1, i):
                    corr = np.sqrt(self.timing[row] / self.timing[col])
                    cov[row, col] = cov[col, row] = corr

            def cdf_at_look_i(u_i):
                limits = boundaries + [u_i]
                if i == 1:
                    return norm.cdf(limits[0])
                else:
                    return multivariate_normal.cdf(limits, mean=np.zeros(i), cov=cov)

            def root_func(u_i):
                return cdf_at_look_i(u_i) - target_cdf

            try:
                boundary = brentq(root_func, -5, 15)
            except ValueError:
                boundary = np.inf

            boundaries.append(boundary)

        if self.alpha < 1 and boundaries[-1] == np.inf:
            try:
                boundary = brentq(root_func, -50, 50)
                boundaries[-1] = boundary
            except ValueError:
                raise RuntimeError("Could not find a valid final boundary.")

        return boundaries

    def simulate(self, n_sims: int, theta: float = 0.0) -> dict:
        """Simulates trials to estimate the operating characteristics of the design.

        Args:
            n_sims (int): The number of trials to simulate.
            theta (float, optional): The effect size (drift parameter).
                `theta = 0` corresponds to the null hypothesis, and the result
                is the Type I error rate. A non-zero theta corresponds to an
                alternative hypothesis, and the result is the power.
                Defaults to 0.0.

        Returns:
            dict: A dictionary containing the simulation results, including:
                - 'rejection_prob': The overall probability of rejecting the null.
                - 'stopping_dist': The distribution of stopping times.
                - 'expected_info': The expected information at trial conclusion.

        Raises:
            ValueError: If `n_sims` is not a positive integer.
        """
        if n_sims <= 0:
            raise ValueError("Number of simulations must be positive.")

        means = theta * np.array(self.timing)

        cov = np.identity(self.k)
        for i in range(self.k):
            for j in range(i + 1, self.k):
                corr = np.sqrt(self.timing[i] / self.timing[j])
                cov[i, j] = cov[j, i] = corr

        simulated_z = np.random.multivariate_normal(mean=means, cov=cov, size=n_sims)

        stopped_at = np.full(n_sims, self.k + 1, dtype=int)
        rejected = np.zeros(n_sims, dtype=bool)

        for i in range(self.k):
            ongoing_trials = stopped_at == self.k + 1
            stopping_now = simulated_z[:, i] >= self.efficacy_boundaries[i]
            update_mask = ongoing_trials & stopping_now
            stopped_at[update_mask] = i + 1
            rejected[update_mask] = True

        rejection_prob = np.mean(rejected)

        stop_counts = np.bincount(stopped_at, minlength=self.k + 2)[
            1:
        ]
        stopping_dist = stop_counts / n_sims

        info_at_stop = np.array(
            self.timing + [self.timing[-1]]
        )

        trial_stop_info = np.array(
            [self.timing[s - 1] if s <= self.k else self.timing[-1] for s in stopped_at]
        )
        expected_info = np.mean(trial_stop_info)

        return {
            "rejection_prob": rejection_prob,
            "stopping_dist": stopping_dist[: self.k],
            "expected_info": expected_info,
        }
