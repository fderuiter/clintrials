"""Classes and methods to perform general useful statistical routines.

Random Seed Strategy: {stats_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"



import numpy as np
from scipy.stats import norm


class ProbabilityDensitySample:
    """Represents a sample from a probability density function.

    This class provides methods to calculate properties of the distribution,
    such as expectation, variance, CDF, and quantiles.
    """

    def __init__(self, samp, func):
        """Initializes a ProbabilityDensitySample object.

        Args:
            samp (numpy.ndarray): The sample from the distribution.
            func (Callable): A function that takes the sample and returns
                the probabilities.
        """
        self._samp = samp
        self._probs = func(samp)
        self._scale = self._probs.mean()

    def expectation(self, vector):
        """Calculates the expectation of a vector.

        Args:
            vector (numpy.ndarray): The vector for which to calculate the
                expectation.

        Returns:
            float: The expectation of the vector.
        """
        return np.mean(vector * self._probs / self._scale)

    def variance(self, vector):
        """Calculates the variance of a vector.

        Args:
            vector (numpy.ndarray): The vector for which to calculate the
                variance.

        Returns:
            float: The variance of the vector.
        """
        exp = self.expectation(vector)
        exp2 = self.expectation(vector**2)
        return exp2 - exp**2


def log_scale_wald_interval(ratio: float, standard_error: float, alpha: float = 0.05) -> tuple[float, float]:
    """Calculates the Wald confidence interval for a ratio on the log scale.

    Args:
        ratio (float): The estimated ratio.
        standard_error (float): The standard error of the log ratio.
        alpha (float): The significance level.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    z_score = norm.ppf(1 - alpha / 2)
    log_ratio = np.log(ratio)
    lower_bound_log = log_ratio - z_score * standard_error
    upper_bound_log = log_ratio + z_score * standard_error
    return (float(np.exp(lower_bound_log)), float(np.exp(upper_bound_log)))


def log_scale_p_value(ratio: float, standard_error: float) -> float:
    """Calculates the two-sided p-value for a ratio using a Wald test on the log scale.

    Args:
        ratio (float): The estimated ratio.
        standard_error (float): The standard error of the log ratio.

    Returns:
        float: The p-value.
    """
    observed_z = np.log(ratio) / standard_error
    return float(2 * norm.sf(abs(observed_z)))


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
