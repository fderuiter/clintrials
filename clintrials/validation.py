"""Validation functions for the Clinical Trials library."""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union

from clintrials.core.errors import ErrorTemplates


def validate_matching_lengths(**kwargs: Sequence[Any]) -> None:
    """Validates that all provided arrays have the same length.

    Pass arrays as keyword arguments. For example:
    validate_matching_lengths(array1=arr1, array2=arr2)

    Args:
        **kwargs: Arbitrary keyword arguments where keys are the names
            of the arrays and values are the arrays themselves.

    Returns:
        None

    Raises:
        ValueError: If any array does not match the length of the first array.
    """
    if not kwargs:
        return

    iterator = iter(kwargs.items())
    first_name, first_arr = next(iterator)
    expected_len = len(first_arr)

    for name, arr in iterator:
        if len(arr) != expected_len:
            raise ValueError(ErrorTemplates.MATCHING_LENGTHS.format(first_name=first_name, name=name))


def validate_expected_length(array: Sequence[Any], expected_length: int, name: str) -> None:
    """Validates that an array has exactly the expected length.

    Args:
        array (list or numpy.ndarray): The array to validate.
        expected_length (int): The expected length of the array.
        name (str): The name of the parameter being validated, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the array length does not match the expected length.
    """
    if len(array) != expected_length:
        raise ValueError(ErrorTemplates.EXPECTED_LENGTH.format(name=name, expected_length=expected_length))


def validate_bounds(value: Union[float, int], lower: Optional[Union[float, int]], upper: Optional[Union[float, int]], name: str, exclusive: bool = False) -> None:
    """Validates that a numerical value is within the specified bounds.

    Args:
        value (float or int): The numerical value to validate.
        lower (float or int, optional): The lower bound.
        upper (float or int, optional): The upper bound.
        name (str): The name of the parameter, used in the error message.
        exclusive (bool, optional): If True, bounds are exclusive (value > lower and value < upper).
            If False, bounds are inclusive (value >= lower and value <= upper). Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If the value is outside the specified bounds.
    """
    if exclusive:
        if lower is not None and value <= lower:
            raise ValueError(ErrorTemplates.GT.format(name=name, bound=lower))
        if upper is not None and value >= upper:
            raise ValueError(ErrorTemplates.LT.format(name=name, bound=upper))
    else:
        if lower is not None and value < lower:
            raise ValueError(ErrorTemplates.GE.format(name=name, bound=lower))
        if upper is not None and value > upper:
            raise ValueError(ErrorTemplates.LE.format(name=name, bound=upper))


def validate_probability(value: float, name: str, exclusive: bool = False) -> None:
    """Validates that a value is a valid probability between 0 and 1.

    Args:
        value (float): The probability value to validate.
        name (str): The name of the parameter, used in the error message.
        exclusive (bool, optional): If True, probabilities of exactly 0 or 1 are invalid.
            Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If the value is not a valid probability.
    """
    if exclusive:
        if not (0 < value < 1):
            raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))
    else:
        if not (0 <= value <= 1):
            raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))


def validate_positive_integer(value: int, name: str) -> None:
    """Validates that a value is a positive integer.

    Args:
        value (int): The value to validate.
        name (str): The name of the parameter, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the value is not an integer or is less than or equal to zero.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(ErrorTemplates.POSITIVE_INTEGER.format(name=name))

