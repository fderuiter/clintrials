from __future__ import annotations
from typing import Any, Callable

__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'
import logging
import warnings
from collections import OrderedDict
from copy import copy
from functools import wraps
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)

def deprecated(alternative):  # type: ignore
    """Decorator to mark a function, method, or class as deprecated.
    Emits a DeprecationWarning pointing to the `alternative`.

    Args:
        alternative (str): The modern alternative function, method, or class to use.
    """

    def decorator(obj):  # type: ignore
        if isinstance(obj, type):
            orig_init = obj.__init__  # type: ignore

            @wraps(orig_init)
            def new_init(self, *args, **kwargs):  # type: ignore
                warnings.warn(f'{obj.__name__} is deprecated and will be removed in a future version. Use {alternative} instead.', category=DeprecationWarning, stacklevel=2)
                orig_init(self, *args, **kwargs)
            obj.__init__ = new_init  # type: ignore
            return obj
        else:

            @wraps(obj)
            def wrapper(*args, **kwargs):  # type: ignore
                warnings.warn(f'{obj.__name__} is deprecated and will be removed in a future version. Use {alternative} instead.', category=DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)
            return wrapper
    return decorator

def get_logger(name: str=__name__) -> logging.Logger:
    """Gets a logger instance.

    Args:
        name (str, optional): The name of the logger. Defaults to `__name__`.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)

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
            list_of_dicts = [x for x in list_of_dicts if x[key] == val or x[key] == list(val)]
        else:
            list_of_dicts = [x for x in list_of_dicts if x[key] == val]
    return list_of_dicts

def tuple_to_dataframe(row_tuples: Any, index_tuples: Any, column_names: Any=None, index_names: Any=None) -> Any:
    """Creates a pandas DataFrame from row and index tuples.

    Args:
        row_tuples (list[dict]): A list of dictionaries representing the rows.
        index_tuples (list[tuple]): A list of tuples representing the MultiIndex.
        column_names (list[str], optional): The column names. Defaults to None.
        index_names (list[str], optional): The names for the index levels.
            Defaults to None.

    Returns:
        pandas.DataFrame: The resulting DataFrame.
    """
    import pandas as pd
    if not row_tuples:
        df = pd.DataFrame(columns=column_names)
        if index_names:
            df.index = pd.MultiIndex.from_tuples([], names=index_names)
        return df
    i = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
    return pd.DataFrame(row_tuples, index=i)

def _correlated_binary_outcomes_mardia(a: Any, b: Any, c: Any) -> Any:
    """Helper function for `correlated_binary_outcomes`."""
    if a == 0:
        return -c / b
    if b > 0:
        k = 1
    elif b < 0:
        k = -1
    else:
        k = 0
    p = -0.5 * (b + k * np.sqrt(b ** 2 - 4 * a * c))
    r1 = 1.0 * p / a
    r2 = 1.0 * c / p
    r = r2 if r2 > 0 else r1
    return r

def _correlated_binary_outcomes_solve2(mui: Any, muj: Any, psi: Any) -> Any:
    """Helper function for `correlated_binary_outcomes`."""
    if psi == 1:
        return mui * muj
    else:
        a = 1 - psi
        b = 1 - a * (mui + muj)
        c = -psi * (mui * muj)
        muij = _correlated_binary_outcomes_mardia(a, b, c)
    return muij

def correlated_binary_outcomes_from_uniforms(unifs: Any, u: Any, psi: Any) -> Any:
    """Generates correlated binary outcomes from uniform random numbers.

    Args:
        unifs (numpy.ndarray): An array of shape (n, 3) of uniform random
            numbers.
        u (list or tuple): A 2-item list or tuple of event probabilities.
        psi (float): The odds ratio of the binary outcomes.

    Returns:
        numpy.ndarray: A 2D array of paired binary outcomes.
    """
    if unifs.ndim == 2 and unifs.shape[1] == 3:
        u12 = _correlated_binary_outcomes_solve2(u[0], u[1], psi)
        n = unifs.shape[0]
        y = -1 * np.ones(shape=(n, 2))
        y[:, 0] = (unifs[:, 0] < u[0]).astype(int)  # type: ignore
        y[:, 1] = y[:, 0] * (unifs[:, 1] <= u12 / u[0]) + (1 - y[:, 0]) * (unifs[:, 2] <= (u[1] - u12) / (1 - u[0]))  # type: ignore
        return y
    else:
        raise ValueError('unifs must be an n*3 array')
from functools import partial


class Memoize:
    """A class to cache function results with a size limit (LRU)."""

    def __init__(self, f: Callable, maxsize: int = 128) -> None:  # type: ignore
        """Initializes a Memoize object.

        Args:
            f (Callable): The function to memoize.
            maxsize (int): The maximum number of entries to keep in cache.
        """
        self.f = f
        self.maxsize = maxsize
        self.memo = OrderedDict()  # type: ignore

    def _make_hashable(self, obj: Any) -> Any:
        if isinstance(obj, (tuple, list)):
            return tuple(self._make_hashable(e) for e in obj)
        elif isinstance(obj, dict):
            return frozenset((k, self._make_hashable(v)) for k, v in obj.items())
        elif isinstance(obj, (int, float, str, bool, frozenset, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return str(id(obj))
        else:
            return str(obj)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the memoized function.

        Args:
            *args: The arguments to the function.
            **kwargs: The keyword arguments to the function.

        Returns:
            The result of the function call.
        """
        # Create a hashable representation of the kwargs
        cache_key = (self._make_hashable(args), self._make_hashable(kwargs))
        if cache_key not in self.memo:
            if len(self.memo) >= self.maxsize:
                self.memo.popitem(last=False)
            self.memo[cache_key] = self.f(*args, **kwargs)
        else:
            # Move to the end to show it was recently used
            self.memo.move_to_end(cache_key)
        return self.memo[cache_key]

    def __get__(self, instance: Any, owner: Any) -> Any:
        """Support instance methods."""
        return partial(self, instance)

class ParameterSpace:
    """A class to handle combinations of parameters in simulations."""

    def __init__(self) -> None:
        """Initializes a ParameterSpace object."""
        self.vals_map = OrderedDict()  # type: ignore

    def add(self, label: Any, values: Any) -> Any:
        """Adds a parameter and its possible values to the space.

        Args:
            label (str): The name of the parameter.
            values (list): A list of possible values for the parameter.
        """
        self.vals_map[label] = values

    def get_cyclical_iterator(self, limit: Any=-1) -> Any:
        """Gets a cyclical iterator for the parameter space.

        Args:
            limit (int, optional): The maximum number of iterations.
                -1 for infinite. Defaults to -1.

        Returns:
            _ParameterSpaceIter: An iterator for the parameter space.
        """
        return _ParameterSpaceIter(self, limit)

    def keys(self) -> Any:
        """Gets the names of the parameters.

        Returns:
            list: A list of parameter names.
        """
        return self.vals_map.keys()

    def dimensions(self) -> Any:
        """Gets the number of values for each parameter.

        Returns:
            numpy.ndarray: An array of the number of values for each
                parameter.
        """
        return np.array([len(y) for x, y in self.vals_map.items()])

    def size(self) -> Any:
        """Gets the total size of the parameter space.

        Returns:
            int: The total number of parameter combinations.
        """
        return np.prod(self.dimensions())

    def __getitem__(self, key: Any) -> Any:
        """Gets the values for a given parameter.

        Args:
            key (str): The name of the parameter.

        Returns:
            list: The list of values for the parameter.
        """
        return self.vals_map[key]

class _ParameterSpaceIter:
    """An iterator for the ParameterSpace class."""

    def __init__(self, parameter_space: Any, limit: Any) -> None:
        """Initializes a _ParameterSpaceIter object."""
        self.limit = limit
        self.cursor = 0
        self.vals_map = copy(parameter_space.vals_map)
        self.labels = list(self.vals_map.keys())
        num_options = []
        for label in self.labels:
            num_options.append(len(parameter_space[label]))
        self.paths = list(product(*[range(x) for x in num_options]))

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        if 0 < self.limit <= self.cursor:
            raise StopIteration()
        i = self.cursor % len(self.paths)
        path = self.paths[i]
        param_map = {}
        assert len(path) == len(self.labels)
        for j, label in enumerate(self.labels):
            param_map[label] = self.vals_map[label][path[j]]
        self.cursor += 1
        return param_map
    next = __next__

from collections.abc import Iterable

__all__ = [
    "get_logger",
    "filter_list_of_dicts",
    "tuple_to_dataframe",
    "correlated_binary_outcomes_from_uniforms",
    "Memoize",
    "ParameterSpace",
    "to_1d_list_gen",
    "to_1d_list",
    "atomic_to_json",
    "iterable_to_json"
]

def to_1d_list_gen(x):  # type: ignore
    """Yield items of a nested list as a 1D generator."""
    if isinstance(x, list):
        for y in x:
            yield from to_1d_list_gen(y)  # type: ignore
    else:
        yield x

def to_1d_list(x):  # type: ignore
    """Convert a nested list into a 1D list."""
    return list(to_1d_list_gen(x))  # type: ignore

def atomic_to_json(obj):  # type: ignore
    """Convert an atomic numpy object to a JSON-serializable type."""
    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def iterable_to_json(obj):  # type: ignore
    """Convert an iterable object to a JSON-serializable list."""
    if isinstance(obj, Iterable):
        return [atomic_to_json(x) for x in obj]  # type: ignore
    else:
        return atomic_to_json(obj)  # type: ignore
