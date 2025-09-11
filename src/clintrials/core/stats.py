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
        x: A list of sample observations.

    Returns:
        A numpy array containing the bootstrap sample.
    """

    return np.random.choice(x, size=len(x), replace=1)


def density(x, n_points=100, covariance_factor=0.25):
    """Calculates and plots an approximate density function from a sample.

    Args:
        x: A list of sample observations.
        n_points: The number of points to estimate in the density function.
        covariance_factor: The covariance factor for the gaussian_kde
            function.
    """

    d = gaussian_kde(x)
    xs = np.linspace(min(x), max(x), n_points)
    d.covariance_factor = lambda: covariance_factor
    d._compute_covariance()
    plt.plot(xs, d(xs))
    plt.show()


def beta_like_normal(mu, sigma):
    """Finds Beta distribution parameters that match a Normal distribution.

    Given X ~ N(mu, sigma^2), this function finds alpha and beta such that
    Y ~ Beta(alpha, beta) has E[Y] = E[X] and Var[Y] = Var[X].

    This is useful for estimating the effective sample size of a normal
    prior, as the effective sample size of Beta(a, b) is a + b.

    Args:
        mu: The mean of the Normal distribution.
        sigma: The standard deviation of the Normal distribution.

    Returns:
        A tuple containing the alpha and beta parameters of the Beta
        distribution.
    """

    alpha = (mu / sigma) ** 2 * (1 - mu) - mu
    beta = ((1 - mu) / mu) * alpha
    return alpha, beta


def or_test(a, b, c, d, ci_alpha=0.05):
    """Calculates the odds ratio and confidence interval for a 2x2 table.

    Args:
        a: Count of positive exposure, positive outcome.
        b: Count of positive exposure, negative outcome.
        c: Count of negative exposure, positive outcome.
        d: Count of negative exposure, negative outcome.
        ci_alpha: The significance level for the confidence interval.

    Returns:
        An ordered dictionary containing the odds ratio, standard error of the
        log odds ratio, and the confidence interval.
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
    """Performs a chi-squared test for association between two lists.

    Args:
        x: The first list of observations.
        y: The second list of observations.
        x_positive_value: The value in `x` that corresponds to a positive
            event. Defaults to 1.
        y_positive_value: The value in `y` that corresponds to a positive
            event. Defaults to 1.
        ci_alpha: The significance level for the odds ratio confidence
            interval.

    Returns:
        An ordered dictionary containing the test statistic, p-value, and
        degrees of freedom. If the data is 2x2, it also includes odds
        ratio information.
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
    """A class for working with samples from a probability density function."""

    def __init__(self, samp, func):
        """Initializes the ProbabilityDensitySample.

        Args:
            samp: A sample from the probability distribution.
            func: A function to calculate the probabilities of the sample.
        """
        self._samp = samp
        self._probs = func(samp)
        self._scale = self._probs.mean()

    def expectation(self, vector):
        """Calculates the expectation of a vector.

        Args:
            vector: The vector for which to calculate the expectation.

        Returns:
            The expectation of the vector.
        """
        return np.mean(vector * self._probs / self._scale)

    def variance(self, vector):
        """Calculates the variance of a vector.

        Args:
            vector: The vector for which to calculate the variance.

        Returns:
            The variance of the vector.
        """
        exp = self.expectation(vector)
        exp2 = self.expectation(vector**2)
        return exp2 - exp**2

    def cdf(self, i, y):
        """Calculates the cumulative density of a parameter.

        This method calculates the probability that the parameter at a given
        position is less than a certain value.

        Args:
            i: The position of the parameter.
            y: The value to compare against.

        Returns:
            The cumulative density.
        """
        return self.expectation(self._samp[:, i] < y)

    def quantile(self, i, p, start_value=0.1):
        """Calculates the quantile of a parameter.

        This method finds the value of the parameter at a given position for
        which a certain proportion of the probability mass is in the left tail.

        Args:
            i: The position of the parameter.
            p: The proportion of the probability mass.
            start_value: The starting value for the solver.

        Returns:
            The value of the quantile.
        """
        return fsolve(lambda z: self.cdf(i, z) - p, start_value)[0]

    def cdf_vector(self, vector, y):
        """Calculates the cumulative density of a sample vector.

        Args:
            vector: The sample vector.
            y: The value to compare against.

        Returns:
            The cumulative density.
        """
        return self.expectation(vector < y)

    def quantile_vector(self, vector, p, start_value=0.1):
        """Calculates the quantile of a sample vector.

        Args:
            vector: The sample vector.
            p: The proportion of the probability mass.
            start_value: The starting value for the solver.

        Returns:
            The value of the quantile.
        """
        return fsolve(lambda z: self.cdf_vector(vector, z) - p, start_value)[0]
