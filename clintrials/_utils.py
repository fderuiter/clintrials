# SPDX-License-Identifier: MIT

"""Private utility functions and helper classes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Dict

import numpy as np


def filter_list_of_dicts(list_of_dicts: Any, filter_dict: Any) -> Any:
    """Filters a list of dictionaries based on a filter dictionary.

    Args:
        list_of_dicts (list[dict]): The list of dictionaries to filter.
        filter_dict (dict): A dictionary of key-value pairs to filter by.

    Returns:
        list[dict]: The filtered list of dictionaries.
    """
    for key, val in filter_dict.items():
        if isinstance(val, tuple):
            list_of_dicts = [
                x for x in list_of_dicts if x[key] == val or x[key] == list(val)
            ]
        else:
            list_of_dicts = [x for x in list_of_dicts if x[key] == val]
    return list_of_dicts


def to_1d_list_gen(x: Any) -> Any:
    """Yield items of a nested list as a 1D generator."""
    if isinstance(x, list):
        for y in x:
            yield from to_1d_list_gen(y)
    else:
        yield x


def to_1d_list(x: Any) -> Any:
    """Convert a nested list into a 1D list."""
    return list(to_1d_list_gen(x))


def atomic_to_json(obj: Any) -> Any:
    """Convert an atomic numpy object to a JSON-serializable type."""
    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def iterable_to_json(obj: Any) -> Any:
    """Convert an iterable object to a JSON-serializable list."""
    if isinstance(obj, Iterable):
        return [atomic_to_json(x) for x in obj]
    else:
        return atomic_to_json(obj)


def filter_kwargs_for_callable(
    func: Callable[..., Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Filters kwargs so that only those accepted by func are returned, unless func accepts ``**kwargs``."""
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return kwargs

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        return kwargs

    filtered = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
    return filtered
