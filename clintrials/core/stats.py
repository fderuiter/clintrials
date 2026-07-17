"""Classes and methods to perform general useful statistical routines.

Random Seed Strategy: {stats_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import logging
from collections import OrderedDict

import numpy as np
from scipy.stats import chi2, norm


def calc_pearson_chi_square(observed, expected, df):
    """Calculates Pearson's chi-squared test statistics and p-values.

    Args:
        observed (numpy.ndarray): The observed frequencies.
        expected (numpy.ndarray): The expected frequencies.
        df (int): The degrees of freedom.

    Returns:
        tuple[float or numpy.ndarray, float or numpy.ndarray]: The test statistic(s)
            and p-value(s).
    """
    test_stat = ((observed - expected) ** 2 / expected)
    if isinstance(test_stat, np.ndarray) and test_stat.ndim > 1:
        test_stat = test_stat.sum(axis=1)
    elif isinstance(test_stat, np.ndarray):
        test_stat = test_stat.sum()
    else:
        test_stat = np.sum(test_stat)
    p = 1 - chi2.cdf(test_stat, df)
    return test_stat, p


def correlation_ci(
    r=None, n=None, samples=None, weights=None, alpha=0.05, method="fisher"
):
    """Calculates a confidence interval for the correlation coefficient.

    Args:
        r (float, optional): The sample correlation coefficient. Required if
            method is 'fisher'.
        n (int, optional): The sample size. Required if method is 'fisher'.
        samples (list or numpy.ndarray, optional): Posterior samples of the
            correlation coefficient. Required if method is 'bayes'.
        weights (list or numpy.ndarray, optional): Weights for the samples,
            useful for importance sampling. Defaults to None.
        alpha (float, optional): The significance level for the confidence
            interval. Defaults to 0.05.
        method (str, optional): The method to use for calculating the
            confidence interval ('fisher' or 'bayes'). Defaults to 'fisher'.

    Returns:
        numpy.ndarray: An array containing the lower bound, mean/point
            estimate, and upper bound of the correlation coefficient.

    Raises:
        ValueError: If required arguments for the chosen method are missing,
            if n < 4 for the 'fisher' method, or if r is not in [-1, 1].
    """
    if method == "fisher":
        if r is None or n is None:
            raise ValueError("r and n are required for method='fisher'")
        if n < 4:
            raise ValueError("n must be at least 4 for Fisher z-transform")
        if not (-1 <= r <= 1):
            raise ValueError("r must be between -1 and 1")

        if r == 1.0:
            return np.array([1.0, 1.0, 1.0])
        if r == -1.0:
            return np.array([-1.0, -1.0, -1.0])

        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = norm.ppf(1 - alpha / 2)
        z_low = z - z_crit * se
        z_high = z + z_crit * se
        return np.array([np.tanh(z_low), r, np.tanh(z_high)])

    elif method == "bayes":
        if samples is None:
            raise ValueError("samples are required for method='bayes'")
        samples = np.array(samples)
        if weights is None:
            return np.array(
                [
                    np.quantile(samples, alpha / 2),
                    np.mean(samples),
                    np.quantile(samples, 1 - alpha / 2),
                ]
            )
        else:
            weights = np.array(weights)
            # Normalize weights
            w = weights / np.sum(weights)
            # Weighted mean
            mean = np.sum(samples * w)
            # Weighted quantiles
            sorter = np.argsort(samples)
            samples = samples[sorter]
            w = w[sorter]
            cumulative_w = np.cumsum(w)
            low = np.interp(alpha / 2, cumulative_w, samples)
            high = np.interp(1 - alpha / 2, cumulative_w, samples)
            return np.array([low, mean, high])
    else:
        raise ValueError("method must be either 'fisher' or 'bayes'")


def bootstrap(x, rng):
    """Creates a bootstrap sample from a list of observations.

    Args:
        x (list): A list of sample observations.
        rng (numpy.random.Generator): A random number generator instance.

    Returns:
        numpy.ndarray: A bootstrap sample of the same size as `x`.
    """
    return rng.choice(x, size=len(x), replace=True)


def beta_like_normal(mu, sigma):
    """Calculates beta distribution parameters that match a normal distribution.

    Given a normal distribution with mean `mu` and standard deviation `sigma`,
    this function finds the `alpha` and `beta` parameters of a beta
    distribution such that the two distributions have the same mean and
    variance.

    This is useful for estimating the effective sample size of a normal prior,
    using the principle that the effective sample size of Beta(a, b) is a+b.

    Args:
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.

    Returns:
        tuple[float, float]: A tuple containing the `alpha` and `beta`
            parameters.
    """
    alpha = (mu / sigma) ** 2 * (1 - mu) - mu
    beta = ((1 - mu) / mu) * alpha
    return alpha, beta


def or_test(a, b, c, d, ci_alpha=0.05):
    """Calculates the odds ratio and its confidence interval.

    Args:
        a (int): The number of observations with positive exposure and
            positive outcome.
        b (int): The number of observations with positive exposure and
            negative outcome.
        c (int): The number of observations with negative exposure and
            positive outcome.
        d (int): The number of observations with negative exposure and
            negative outcome.
        ci_alpha (float, optional): The significance level for the confidence
            interval. Defaults to 0.05.

    Returns:
        collections.OrderedDict: A dictionary containing the odds ratio,
            log(OR) standard error, confidence interval, and alpha.
    """
    abcd = [a, b, c, d]
    to_return = OrderedDict()
    to_return["ABCD"] = abcd

    if np.any(np.array(abcd) < 0):
        logging.error("Negative event count. Garbage!")
    elif np.any(np.array(abcd) == 0):
        logging.info("At least one event count was zero. Added one to all counts.")
        abcd = np.array(abcd) + 1
        a, b, c, d = abcd

    odds_ratio = 1.0 * (a * d) / (c * b)
    log_or_se = np.sqrt(sum(1.0 / np.array(abcd)))
    ci_scalars = norm.ppf([ci_alpha / 2, 1 - ci_alpha / 2])
    or_ci = np.exp(np.log(odds_ratio) + ci_scalars * log_or_se)

    to_return["OR"] = odds_ratio
    to_return["Log(OR) SE"] = log_or_se
    to_return["OR CI"] = list(or_ci)

    to_return["Alpha"] = ci_alpha
    return to_return


def chi_squ_test(x, y, x_positive_value=None, y_positive_value=None, ci_alpha=0.05):
    """Performs a chi-squared test for association between two variables.

    Args:
        x (list): The first variable.
        y (list): The second variable.
        x_positive_value (object, optional): The value in `x` that
            corresponds to a positive event. Defaults to 1.
        y_positive_value (object, optional): The value in `y` that
            corresponds to a positive event. Defaults to 1.
        ci_alpha (float, optional): The significance level for the confidence
            interval of the odds ratio (if applicable). Defaults to 0.05.

    Returns:
        collections.OrderedDict: A dictionary containing the test statistic,
            p-value, degrees of freedom, and odds ratio results (for 2x2
            tables).
    """
    obs_list = []
    exp_list = []
    x_set = set(x)
    y_set = set(y)
    for x_case in x_set:
        x_matches = [z == x_case for z in x]
        for y_case in y_set:
            y_matches = [z == y_case for z in y]
            obs = sum(np.array(x_matches) & np.array(y_matches))
            exp = 1.0 * sum(x_matches) * sum(y_matches) / len(x)
            obs_list.append(obs)
            exp_list.append(exp)
    num_df = (len(x_set) - 1) * (len(y_set) - 1)
    sum_oe, p = calc_pearson_chi_square(np.array(obs_list), np.array(exp_list), num_df)
    to_return = OrderedDict([("TestStatistic", sum_oe), ("p", p), ("Df", num_df)])

    if len(x_set) == 2 and len(y_set) == 2:
        x = np.array(x)
        y = np.array(y)
        if not x_positive_value:
            x_positive_value = 1
        if not y_positive_value:
            y_positive_value = 1
        x_pos_val, y_pos_val = x_positive_value, y_positive_value
        a, b, c, d = (
            sum((x == x_pos_val) & (y == y_pos_val)),
            sum((x == x_pos_val) & (y != y_pos_val)),
            sum((x != x_pos_val) & (y == y_pos_val)),
            sum((x != x_pos_val) & (y != y_pos_val)),
        )
        to_return["Odds"] = or_test(a, b, c, d, ci_alpha=ci_alpha)
    else:
        # There's no reason why the OR logic could not be calculated for each combination pair
        # in x and y, but it's more work so leave it for now.
        pass

    return to_return


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

    def cdf(self, i, y):
        """Calculates the cumulative density function (CDF).

        This method calculates the probability that the parameter at a given
        position is less than a certain value.

        Args:
            i (int): The index of the parameter in the sample.
            y (float): The value to compare against.

        Returns:
            float: The cumulative density.
        """
        return self.expectation(self._samp[:, i] < y)

    def quantile(self, i, p, start_value=0.1):
        """Calculates the quantile.

        This method finds the value of the parameter at a given position for
        which a certain proportion of the probability mass is in the
        left-tail.

        Args:
            i (int): The index of the parameter in the sample.
            p (float): The desired probability (between 0 and 1).
            start_value (float, optional): The starting value for the solver.
                Defaults to 0.1.

        Returns:
            float: The quantile value.
        """
        samples = self._samp[:, i]
        weights = self._probs
        sorter = np.argsort(samples)
        samples = samples[sorter]
        w = weights[sorter]
        w = w / np.sum(w)
        cumulative_w = np.cumsum(w)
        return float(np.interp(p, cumulative_w, samples))

    def cdf_vector(self, vector, y):
        """Calculates the CDF for a vector.

        This method calculates the probability that the given vector is less
        than a certain value.

        Args:
            vector (numpy.ndarray): The vector.
            y (float): The value to compare against.

        Returns:
            float: The cumulative density.
        """
        return self.expectation(vector < y)

    def quantile_vector(self, vector, p, start_value=0.1):
        """Calculates the quantile for a vector.

        This method finds the value for which a certain proportion of the
        probability mass of the given vector is in the left-tail.

        Args:
            vector (numpy.ndarray): The vector.
            p (float): The desired probability (between 0 and 1).
            start_value (float, optional): The starting value for the solver.
                Defaults to 0.1.

        Returns:
            float: The quantile value.
        """
        samples = np.array(vector)
        weights = self._probs
        sorter = np.argsort(samples)
        samples = samples[sorter]
        w = weights[sorter]
        w = w / np.sum(w)
        cumulative_w = np.cumsum(w)
        return float(np.interp(p, cumulative_w, samples))


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
