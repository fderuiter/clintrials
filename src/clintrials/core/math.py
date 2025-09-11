__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


""" Common, useful functions in the statistics and mathematics of clinical trials. """

import numpy as np


def inverse_logit(x):
    """Return the inverse logit of ``x``.

    The inverse logit is ``1 / (1 + exp(-x))``.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Result of the inverse logit.

    Examples
    --------
    >>> inverse_logit(0)
    0.5
    """

    return 1 / (1 + np.exp(-x))


# Two-parameter link functions used in CRM-style designs
# They are written in pairs and all use the same call signature.
# They take their lead from the same in the dfcrm R-package.
def empiric(x, a0=None, beta=0):
    """Calculates the empiric function value.

    The empiric function is defined as :math:`x^{e^\\beta}`.

    Args:
        x: The input value.
        a0: The intercept parameter (ignored, for signature compatibility).
        beta: The slope parameter.

    Returns:
        The result of the empiric function.

    Examples:
        >>> import math
        >>> empiric(0.5, beta=math.log(2))
        0.25
    """
    beta = np.clip(beta, -10, 10)
    return x ** np.exp(beta)


def inverse_empiric(x, a0=0, beta=0):
    """Calculates the inverse empiric function value.

    The inverse empiric function is defined as :math:`x^{e^{-\\beta}}`.

    Note:
        This function is the inverse of :func:`clintrials.core.math.empiric`.

    Args:
        x: The input value.
        a0: The intercept parameter (ignored, for signature compatibility).
        beta: The slope parameter.

    Returns:
        The result of the inverse empiric function.

    Examples:
        >>> import math
        >>> inverse_empiric(0.25, beta=math.log(2))
        0.5
    """
    return x ** np.exp(-beta)


def logistic(x, a0=0, beta=0):
    """Calculates the logistic function value.

    The logistic function is defined as
    :math:`\\frac{1}{1 + e^{-a_0 - e^\\beta x}}`.

    Args:
        x: The input value.
        a0: The intercept parameter.
        beta: The slope parameter.

    Returns:
        The result of the logistic function.

    Examples:
        >>> logistic(0.25, -1, 1)
        0.42057106852688747
    """
    beta = np.clip(beta, -10, 10)
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))


def inverse_logistic(x, a0=0, beta=0):
    """Calculates the inverse logistic function value.

    The inverse logistic function is defined as
    :math:`\\frac{\\log(\\frac{x}{1-x}) - a_0}{e^\\beta}`.

    Note:
        This function is the inverse of :func:`clintrials.core.math.logistic`.

    Args:
        x: The input value.
        a0: The intercept parameter.
        beta: The slope parameter.

    Returns:
        The result of the inverse logistic function.

    Examples:
        >>> round(inverse_logistic(0.42057106852688747, -1, 1), 2)
        0.25
    """
    return (np.log(x / (1 - x)) - a0) / np.exp(beta)


def hyperbolic_tan(x, a0=0, beta=0):
    """Calculates the hyperbolic tangent function value.

    Args:
        x: The input value.
        a0: The intercept parameter (ignored).
        beta: The slope parameter.

    Returns:
        The result of the hyperbolic tangent function.
    """
    return ((np.tanh(x) + 1) / 2) ** np.exp(beta)


def inverse_hyperbolic_tan(x, a0=0, beta=0):
    """Calculates the inverse hyperbolic tangent function value.

    Args:
        x: The input value.
        a0: The intercept parameter (ignored).
        beta: The slope parameter.

    Returns:
        The result of the inverse hyperbolic tangent function.
    """
    return np.arctanh(2 * x ** np.exp(-beta) - 1)


def logit1(x, a0=3, beta=0):
    """Calculates the 1-parameter logistic function value.

    The 1-parameter logistic function is defined as
    :math:`\\frac{e^{a_0+\\alpha x}}{1 + e^{a_0+\\alpha x}}`, where
    :math:`\\alpha = e^{\\beta}`.

    Args:
        x: The input value.
        a0: The intercept parameter.
        beta: The slope parameter.

    Returns:
        The result of the 1-parameter logistic function.
    """
    beta = np.clip(beta, -10, 10)
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))


def inverse_logit1(x, a0=3, beta=0):
    """Calculates the inverse 1-parameter logistic function value.

    Note:
        This function is the inverse of :func:`clintrials.core.math.logit1`.

    Args:
        x: The input value.
        a0: The intercept parameter.
        beta: The slope parameter.

    Returns:
        The result of the inverse 1-parameter logistic function.
    """
    return (np.log(x / (1 - x)) - a0) / np.exp(beta)
