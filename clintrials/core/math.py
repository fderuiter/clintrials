__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


""" Common, useful functions in the statistics and mathematics of clinical trials. """

import numpy as np


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
def empiric(x, a0=None, beta=0):
    """Calculates the empiric function value.

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

    beta = np.clip(beta, -10, 10)
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


def logistic(x, a0=0, beta=0):
    """Calculates the logistic function value.

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
    beta = np.clip(beta, -10, 10)
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))


def inverse_logistic(x, a0=0, beta=0):
    """Calculates the inverse logistic function value.

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
    beta = np.clip(beta, -10, 10)
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
    """
    return np.arctanh(2 * x ** np.exp(-beta) - 1)


def logit1(x, a0=3, beta=0):
    """Calculates the 1-parameter logistic function value.

    This is a logistic function, typically used with a single parameter `beta`,
    and a fixed intercept `a0`. The formula is:
    1 / (1 + exp(-a0 - exp(beta) * x))

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. Defaults to 3.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The logistic function value.
    """
    beta = np.clip(beta, -10, 10)
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))


def inverse_logit1(x, a0=3, beta=0):
    """Calculates the inverse 1-parameter logistic function value.

    This function is the inverse of `logit1`.

    Args:
        x (float): The input value.
        a0 (float, optional): Intercept parameter. Defaults to 3.
        beta (float, optional): The slope parameter. Defaults to 0.

    Returns:
        float: The inverse logistic function value.
    """
    beta = np.clip(beta, -10, 10)
    return (np.log(x / (1 - x)) - a0) / np.exp(beta)
