"""Common, useful functions in the statistics and mathematics of clinical trials.

Random Seed Strategy: {math_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

import numpy as np

from clintrials.core.registry import CORE_REGISTRY, inject_docs


def logit(p):
    """Calculates the logit of a probability.

    The probability is silently clipped to [1e-7, 1 - 1e-7] to prevent log(0).

    Args:
        p (float or numpy.ndarray): The probability.

    Returns:
        float or numpy.ndarray: The logit.
    """
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def bernoulli_likelihood(p, y, log=False):
    """Calculates the Bernoulli likelihood or log-likelihood.

    Args:
        p (float or numpy.ndarray): Probability of success.
        y (int or numpy.ndarray): Observed outcome(s) (1 or 0).
        log (bool, optional): If True, returns the log-likelihood. Defaults to False.

    Returns:
        float or numpy.ndarray: The likelihood or log-likelihood.
    """
    p = np.clip(p, 1e-15, 1 - 1e-15)
    log_l = y * np.log(p) + (1 - y) * np.log(1 - p)
    if log:
        return log_l
    else:
        return np.exp(np.clip(log_l, -700, 700))


def inverse_logit(x):
    """Calculates the inverse logit of a number.

    The inverse logit is defined as 1 / (1 + exp(-x)).

    Args:
        x (float): The number to apply the inverse logit function to.

    Returns:
        float: The result of the inverse logit function.

    Examples:
        >>> inverse_logit(0)
        0.5
    """
    return 1 / (1 + np.exp(-x))


# Two-parameter link functions used in CRM-style designs
# They are written in pairs and all use the same call signature.
# They take their lead from the same in the dfcrm R-package.
@inject_docs()
def empiric(x, a0=None, beta=0):
    """Calculates the empiric function value. Beta values are silently clipped to the range [{math_clip_beta_min}, {math_clip_beta_max}] to prevent overflow.

    The formula is: x^(e^beta)

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. This parameter is ignored
            but exists to match the call signature of other link functions.
            Defaults to None.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The empiric function value.

    Examples:
        >>> import math
        >>> empiric(0.5, beta=math.log(2))
        0.25
    """
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return x ** np.exp(beta)


def inverse_empiric(x, a0=0, beta=0):
    """Calculates the inverse empiric function value.

    This function is the inverse of `empiric`. The formula is: x^(e^(-beta))

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. This parameter is ignored
            but exists to match the call signature of other link functions.
            Defaults to 0.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The inverse empiric function value.

    Examples:
        >>> import math
        >>> inverse_empiric(0.25, beta=math.log(2))
        0.5
    """
    return x ** np.exp(-beta)


@inject_docs()
def logistic(x, a0=0, beta=0):
    """Calculates the logistic function value. Beta values are silently clipped to the range [{math_clip_beta_min}, {math_clip_beta_max}] to prevent overflow.

    The formula is: 1 / (1 + e^(-a0 - e^beta * x))

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. Defaults to 0.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The logistic function value.

    Examples:
        >>> logistic(0.25, -1, 1)
        0.42057106852688747
    """
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))


@inject_docs()
def inverse_logistic(x, a0=0, beta=0):
    """Calculates the inverse logistic function value. Beta values are silently clipped to the range [{math_clip_beta_min}, {math_clip_beta_max}] to prevent overflow.

    This function is the inverse of `logistic`.
    The formula is: (log(x / (1 - x)) - a0) / e^beta

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. Defaults to 0.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The inverse logistic function value.

    Examples:
        >>> round(inverse_logistic(0.42057106852688747, -1, 1), 2)
        0.25
    """
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return (np.log(x / (1 - x)) - a0) / np.exp(beta)


def hyperbolic_tan(x, a0=0, beta=0):
    """Calculates the hyperbolic tangent function.

    The formula is: ((tanh(x) + 1) / 2) ** exp(beta)

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. This parameter is ignored
            but exists to match the call signature of other link functions.
            Defaults to 0.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The result of the hyperbolic tangent function.

    Examples:
        >>> hyperbolic_tan(0.5, beta=1)
        0.559...
    """
    return ((np.tanh(x) + 1) / 2) ** np.exp(beta)


def inverse_hyperbolic_tan(x, a0=0, beta=0):
    """Calculates the inverse hyperbolic tangent function.

    This function is the inverse of `hyperbolic_tan`.

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. This parameter is ignored
            but exists to match the call signature of other link functions.
            Defaults to 0.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The result of the inverse hyperbolic tangent function.

    Examples:
        >>> inverse_hyperbolic_tan(0.559..., beta=1)
        0.5
    """
    return np.arctanh(2 * x ** np.exp(-beta) - 1)


@inject_docs()
def logit1(x, a0=3, beta=0):
    """Calculates the 1-parameter logistic function value. Beta values are silently clipped to the range [{math_clip_beta_min}, {math_clip_beta_max}] to prevent overflow.

    This is a logistic function, typically used with a single parameter `beta`,
    and a fixed intercept `a0`. The formula is:
    1 / (1 + exp(-a0 - exp(beta) * x))

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. Defaults to 3.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The logistic function value.

    Examples:
        >>> logit1(0.5, beta=1)
        0.972...
    """
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))


@inject_docs()
def inverse_logit1(x, a0=3, beta=0):
    """Calculates the inverse 1-parameter logistic function value. Beta values are silently clipped to the range [{math_clip_beta_min}, {math_clip_beta_max}] to prevent overflow.

    This function is the inverse of `logit1`.

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. Defaults to 3.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The inverse logistic function value.

    Examples:
        >>> inverse_logit1(0.972..., beta=1)
        0.5
    """
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return (np.log(x / (1 - x)) - a0) / np.exp(beta)


def association_to_correlation(psi):
    """Converts an association parameter to a correlation coefficient.

    The formula is: (e^psi - 1) / (e^psi + 1)

    Args:
        psi (float or numpy.ndarray): The association parameter.

    Returns:
        float or numpy.ndarray: The correlation coefficient.
    """
    return (np.exp(psi) - 1) / (np.exp(psi) + 1)


def fgm_joint_prob(a, b, p1, p2, psi):
    """Calculates the joint probability of two Bernoulli variables using an FGM copula.

    Args:
        a (int or numpy.ndarray): The outcome of the first variable (1 or 0).
        b (int or numpy.ndarray): The outcome of the second variable (1 or 0).
        p1 (float or numpy.ndarray): The marginal probability of the first variable.
        p2 (float or numpy.ndarray): The marginal probability of the second variable.
        psi (float or numpy.ndarray): The association parameter.

    Returns:
        float or numpy.ndarray: The joint probability.
    """
    prob = p1**a * (1 - p1) ** (1 - a) * p2**b * (1 - p2) ** (1 - b)
    prob += (
        (-1) ** (a + b)
        * p1
        * (1 - p1)
        * p2
        * (1 - p2)
        * association_to_correlation(psi)
    )
    return prob


# Inject module-level docstring
if __doc__:
    __doc__ = __doc__.format(**CORE_REGISTRY)
