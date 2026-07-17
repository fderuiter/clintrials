"""Module for Group Sequential Designs (GSDs).

Random Seed Strategy: {gsd_seed_strategy}
"""

from typing import Callable, List

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from clintrials.core.registry import CORE_REGISTRY, inject_docs
from clintrials.utils import deprecated


@inject_docs()
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


from clintrials.core.protocol import Protocol


class GroupSequentialDesign(Protocol):
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
        from clintrials.validation import (
            validate_positive_integer,
            validate_probability,
        )

        validate_probability(alpha, "alpha", exclusive=True)
        validate_positive_integer(k, "k")

        self.k = k
        self.alpha = alpha
        self.sfu = sfu
        self.timing = timing if timing is not None else np.linspace(1 / k, 1, k)

        from clintrials.validation import validate_expected_length

        validate_expected_length(self.timing, k, "timing")
        if any(self.timing[i] >= self.timing[i + 1] for i in range(k - 1)):
            raise ValueError("Timing must be strictly increasing.")
        if self.timing[-1] != 1.0:
            raise ValueError("The last element of timing must be 1.0.")

        self.efficacy_boundaries = self._compute_efficacy_boundaries()
        self.reset()

    def reset(self):
        """Reset the group sequential design state."""
        self._stage = 0
        self._stopped = False
        self._rejected = False
        self._z_scores = []
        self._information = []

    def update(self, z_score: float, info: float = None):
        """Update the trial state with the latest test statistic."""
        if self._stopped:
            return
        self._stage += 1
        self._z_scores.append(z_score)

        if info is not None:
            self._information.append(info)
        else:
            self._information.append(self.timing[self._stage - 1])

        if z_score >= self.efficacy_boundaries[self._stage - 1]:
            self._stopped = True
            self._rejected = True
        elif self._stage >= self.k:
            self._stopped = True

    def has_more(self):
        """Check if the trial should continue to the next stage."""
        return not self._stopped

    def report(self):
        """Generate a report of the trial state and results."""
        from collections import OrderedDict

        from clintrials.utils import atomic_to_json, iterable_to_json

        report = OrderedDict()
        report["Stage"] = atomic_to_json(self._stage)
        report["Stopped"] = atomic_to_json(self._stopped)
        report["Rejected"] = atomic_to_json(self._rejected)
        report["ZScores"] = iterable_to_json(self._z_scores)
        report["Information"] = iterable_to_json(self._information)
        return report

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
                    return multivariate_normal.cdf(
                        limits,
                        mean=np.zeros(i),
                        cov=cov,
                        maxpts=CORE_REGISTRY["gsd_maxpts"],
                        abseps=CORE_REGISTRY["gsd_abseps"],
                    )

            def root_func(u_i):
                return cdf_at_look_i(u_i) - target_cdf

            try:
                boundary = brentq(
                    root_func,
                    CORE_REGISTRY["gsd_brentq_first_min"],
                    CORE_REGISTRY["gsd_brentq_first_max"]
                )
            except ValueError:
                boundary = np.inf

            boundaries.append(boundary)

        if self.alpha < 1 and boundaries[-1] == np.inf:
            try:
                boundary = brentq(
                    root_func,
                    CORE_REGISTRY["gsd_brentq_second_min"],
                    CORE_REGISTRY["gsd_brentq_second_max"]
                )
                boundaries[-1] = boundary
            except ValueError:
                raise RuntimeError("Could not find a valid final boundary.")

        return boundaries

    @deprecated(alternative="run(..., method='bulk')")
    def simulate(self, n_sims: int, theta: float = 0.0):
        """Legacy method for backward compatibility."""
        # Calling run without a seed keeps it stochastic, but we can just use the protocol's runner.
        return self.run(n_sims=n_sims, method="bulk", theta=theta)

# Inject module-level docstring
if __doc__:
    __doc__ = __doc__.format(**CORE_REGISTRY)
