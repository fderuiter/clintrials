__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

""" Classes and methods to perform general useful statistical routines. """


import logging
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import chi2, gaussian_kde, norm


def bootstrap(x):
    """Creates a bootstrap sample from a list of observations.

    Args:
        x (list): A list of sample observations.

    Returns:
        numpy.ndarray: A bootstrap sample of the same size as `x`.
    """
    return np.random.choice(x, size=len(x), replace=1)


def density(x, n_points=100, covariance_factor=0.25):
    """Calculates and plots an approximate density function from a sample.

    Args:
        x (list): A list of sample observations.
        n_points (int, optional): The number of points in the density
            function to estimate. Defaults to 100.
        covariance_factor (float, optional): The covariance factor for the
            Gaussian KDE. See `scipy.stats.gaussian_kde` for details.
            Defaults to 0.25.
    """
    d = gaussian_kde(x)
    xs = np.linspace(min(x), max(x), n_points)
    d.covariance_factor = lambda: covariance_factor
    d._compute_covariance()
    plt.plot(xs, d(xs))
    plt.show()


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
    sum_oe = 0.0
    x_set = set(x)
    y_set = set(y)
    for x_case in x_set:
        x_matches = [z == x_case for z in x]
        for y_case in y_set:
            y_matches = [z == y_case for z in y]
            obs = sum(np.array(x_matches) & np.array(y_matches))
            exp = 1.0 * sum(x_matches) * sum(y_matches) / len(x)
            oe = (obs - exp) ** 2 / exp
            sum_oe += oe
    num_df = (len(x_set) - 1) * (len(y_set) - 1)
    p = 1 - chi2.cdf(sum_oe, num_df)
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
            func (callable): A function that takes the sample and returns
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
        return fsolve(lambda z: self.cdf(i, z) - p, start_value)[0]

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
        return fsolve(lambda z: self.cdf_vector(vector, z) - p, start_value)[0]
