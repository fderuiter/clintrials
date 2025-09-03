__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

""" Implementations of simple phase II clinical trial designs. Long, complicated designs belong in own modules. """

from collections import OrderedDict
from itertools import product

import numpy as np
from scipy.stats import beta, binom, chi2


def bayesian_2stage_dich_design(
    theta,
    p0,
    p1,
    N0,
    N1,
    p,
    q,
    prior_a=1,
    prior_b=1,
    labels=["StopAtInterim", "StopAtFinal", "GoAtFinal"],
):
    """Calculates outcome probabilities for a two-stage Bayesian trial.

    This function calculates the probabilities of stopping at the interim
    analysis, stopping at the final analysis, or declaring success at the
    final analysis for a two-stage Bayesian design with a dichotomous
    endpoint.

    Args:
        theta (float): The true efficacy rate.
        p0 (float): The lower bound for the efficacy rate (null hypothesis).
        p1 (float): The upper bound for the efficacy rate (alternative
            hypothesis).
        N0 (int): The number of patients in the first stage.
        N1 (int): The total number of patients in both stages.
        p (float): The posterior probability threshold for stopping at the
            interim analysis.
        q (float): The posterior probability threshold for declaring success
            at the final analysis.
        prior_a (float, optional): The alpha parameter of the beta prior
            distribution for theta. Defaults to 1.
        prior_b (float, optional): The beta parameter of the beta prior
            distribution for theta. Defaults to 1.
        labels (list[str], optional): The labels for the three possible
            outcomes. Defaults to ["StopAtInterim", "StopAtFinal",
            "GoAtFinal"].

    Returns:
        dict: A dictionary mapping the outcome labels to their
            probabilities.
    """
    a, b = prior_a, prior_b
    n0, n1 = zip(*product(range(N0 + 1), range(N1 - N0 + 1)))
    n0, n1 = np.array(n0), np.array(n1)
    lik0 = binom.pmf(n0, n=N0, p=theta)
    lik1 = binom.pmf(n1, n=N1 - N0, p=theta)
    joint_lik = lik0 * lik1
    prob_lt_p0 = beta.cdf(p0, a + n0, b + N0 - n0)
    prob_gt_p1 = 1 - beta.cdf(p1, a + n0 + n1, b + N1 - n0 - n1)
    stop_0 = sum(joint_lik[prob_lt_p0 > p])
    go_1 = sum(joint_lik[~(prob_lt_p0 > p) & (prob_gt_p1 > q)])
    stop_1 = 1 - stop_0 - go_1
    return {labels[0]: stop_0, labels[1]: stop_1, labels[2]: go_1}


def bayesian_2stage_dich_design_df(
    theta,
    p0,
    p1,
    N0,
    N1,
    p,
    q,
    prior_a=1,
    prior_b=1,
    labels=["StopAtInterim", "StopAtFinal", "GoAtFinal"],
):
    """Calculates outcome probabilities and returns a DataFrame.

    This function is similar to `bayesian_2stage_dich_design`, but it
    returns a pandas DataFrame with detailed information about each possible
    outcome.

    Args:
        theta (float): The true efficacy rate.
        p0 (float): The lower bound for the efficacy rate (null hypothesis).
        p1 (float): The upper bound for the efficacy rate (alternative
            hypothesis).
        N0 (int): The number of patients in the first stage.
        N1 (int): The total number of patients in both stages.
        p (float): The posterior probability threshold for stopping at the
            interim analysis.
        q (float): The posterior probability threshold for declaring success
            at the final analysis.
        prior_a (float, optional): The alpha parameter of the beta prior
            distribution for theta. Defaults to 1.
        prior_b (float, optional): The beta parameter of the beta prior
            distribution for theta. Defaults to 1.
        labels (list[str], optional): The labels for the three possible
            outcomes. Defaults to ["StopAtInterim", "StopAtFinal",
            "GoAtFinal"].

    Returns:
        pandas.DataFrame: A DataFrame with detailed information about each
            possible outcome.
    """
    a, b = prior_a, prior_b
    n0, n1 = zip(*product(range(N0 + 1), range(N1 - N0 + 1)))
    n0, n1 = np.array(n0), np.array(n1)
    lik0 = binom.pmf(n0, n=N0, p=theta)
    lik1 = binom.pmf(n1, n=N1 - N0, p=theta)
    joint_lik = lik0 * lik1
    prob_lt_p0 = beta.cdf(p0, a + n0, b + N0 - n0)
    prob_gt_p1 = 1 - beta.cdf(p1, a + n0 + n1, b + N1 - n0 - n1)
    stop_0 = prob_lt_p0 > p
    go_1 = ~stop_0 & (prob_gt_p1 > q)
    stop_1 = ~(stop_0 | go_1)

    dat = OrderedDict()
    dat["Successes0"] = n0
    dat["Pr(theta<p0)"] = prob_lt_p0
    dat["Successes1"] = n1
    dat["TotalSuccesses"] = n0 + n1
    dat["Pr(theta>p1)"] = prob_gt_p1
    dat["Lik"] = joint_lik
    dat[labels[0]] = stop_0
    dat[labels[1]] = stop_1
    dat[labels[2]] = go_1
    import pandas as pd

    return pd.DataFrame(dat)


def chisqu_two_arm_comparison(p0, p1, n, alpha):
    """Performs a chi-squared test for a two-arm comparison.

    This function calculates the power of a chi-squared test to detect a
    difference between two proportions.

    Args:
        p0 (float): The proportion in the first arm.
        p1 (float): The proportion in the second arm.
        n (int): The number of patients per arm.
        alpha (float): The significance level.

    Returns:
        tuple[float, float]: A tuple containing the probability of
            rejecting the null hypothesis and the probability of not
            rejecting it.
    """
    n0, n1 = zip(*list(product(range(n + 1), range(n + 1))))
    n0 = np.array(n0)
    n1 = np.array(n1)
    lik0 = binom.pmf(n0, n, p0)
    lik1 = binom.pmf(n1, n, p1)
    lik = lik0 * lik1
    observed = np.column_stack([n0, n - n0, n1, n - n1])
    success = n0 + n1
    fail = 2 * n - n0 - n1
    expected = np.column_stack([success / 2.0, fail / 2.0, success / 2.0, fail / 2.0])
    test_stat = ((observed - expected) ** 2 / expected).sum(axis=1)
    p = 1 - chi2.cdf(test_stat, 1)
    reject = (p < alpha * 2) & (n0 < n1)
    data = np.column_stack([n0, n1, lik, test_stat, p, reject])
    return sum(data[data[:, 5] == True, 2]), sum(data[data[:, 5] == False, 2])
