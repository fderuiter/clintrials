# SPDX-License-Identifier: MIT

"""Utility functions and helper classes for clinical trials simulations."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

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
        y = np.full((n, 2), -1, dtype=int)
        y[:, 0] = (unifs[:, 0] < u[0]).astype(int)
        y[:, 1] = y[:, 0] * (unifs[:, 1] <= u12 / u[0]) + (1 - y[:, 0]) * (unifs[:, 2] <= (u[1] - u12) / (1 - u[0]))
        return y
    else:
        raise ValueError('unifs must be an n*3 array')
class _HashableArgs:
    """A helper class that wraps unhashable arguments.

    Wraps original arguments and keyword arguments together with their
    pre-processed hashable key representation for use in `lru_cache`.
    """

    def __init__(self, args: Any, kwargs: Any, key: Any) -> None:
        """Initializes a _HashableArgs object.

        Args:
            args (tuple): The original positional arguments.
            kwargs (dict): The original keyword arguments.
            key (tuple): The pre-processed, hashable representation of the arguments.
        """
        self.args = args
        self.kwargs = kwargs
        self.key = key

    def __hash__(self) -> int:
        """Returns the hash of the pre-processed key.

        Returns:
            int: The hash value.
        """
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        """Checks equality against another _HashableArgs object based on their keys.

        Args:
            other (Any): The other object to compare with.

        Returns:
            bool: True if keys are equal, False otherwise.
        """
        if not isinstance(other, _HashableArgs):
            return NotImplemented
        return bool(self.key == other.key)


class Memoize:
    """A class to cache function results with a size limit (LRU)."""

    def __init__(self, f: Optional[Callable[..., Any]] = None, maxsize: int = 128) -> None:
        """Initializes a Memoize object.

        Args:
            f (Callable, optional): The function to memoize. Defaults to None.
            maxsize (int): The maximum number of entries to keep in cache.
        """
        self.f = f
        self.maxsize = maxsize
        self.cache: OrderedDict[Any, Any] = OrderedDict()

    def _make_hashable(self, obj: Any, seen: Any = None) -> Any:
        """Pre-processes complex, unhashable parameters into a hashable structure.

        Performs recursive attribute serialization of custom object instances
        and nested collections, handling numpy arrays and cycle detection.

        Args:
            obj (Any): The object/value to serialize.
            seen (set, optional): A set of object IDs already processed in the
                recursion stack to detect cycles. Defaults to None.

        Returns:
            Any: A hashable, deeply-nested representation of the input.
        """
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return f"<cycle-{obj_id}>"

        is_container = (hasattr(obj, '__dict__') and not callable(obj)) or isinstance(
            obj, (list, tuple, dict, set, frozenset, np.ndarray)
        )
        if is_container:
            seen.add(obj_id)

        try:
            if isinstance(obj, (tuple, list)):
                return tuple(self._make_hashable(e, seen) for e in obj)
            elif isinstance(obj, dict):
                return frozenset((k, self._make_hashable(v, seen)) for k, v in obj.items())
            elif isinstance(obj, (int, float, str, bool, frozenset, type(None))):
                return obj
            elif isinstance(obj, np.ndarray):
                return tuple(self._make_hashable(e, seen) for e in obj.tolist())
            elif isinstance(obj, type):
                return (obj.__module__, obj.__name__)
            elif callable(obj):
                # Process dynamic callable arguments separately from standard data attributes
                # to maintain stable key signatures and prevent serialization errors.
                if hasattr(obj, '__self__') and hasattr(obj, '__func__'):
                    try:
                        self_module = obj.__self__.__class__.__module__
                        self_class = obj.__self__.__class__.__name__
                    except Exception:
                        self_module = None
                        self_class = None
                    return (
                        obj.__class__.__module__,
                        obj.__class__.__name__,
                        self_module,
                        self_class,
                        getattr(obj.__func__, '__module__', None),
                        getattr(obj.__func__, '__qualname__', None)
                    )
                elif hasattr(obj, '__code__'):
                    closure_val: Optional[Any] = None
                    if hasattr(obj, '__closure__') and obj.__closure__ is not None:
                        try:
                            closure_val = tuple(self._make_hashable(cell.cell_contents, seen) for cell in obj.__closure__)
                        except Exception:
                            closure_val = "<unserializable-closure>"
                    return (
                        obj.__class__.__module__,
                        obj.__class__.__name__,
                        getattr(obj, '__module__', None),
                        getattr(obj, '__qualname__', None),
                        closure_val
                    )
                else:
                    return (
                        obj.__class__.__module__,
                        obj.__class__.__name__,
                        getattr(obj, '__module__', None),
                        getattr(obj, '__qualname__', None)
                    )
            elif hasattr(obj, '__dict__'):
                return (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    self._make_hashable(obj.__dict__, seen)
                )
            elif hasattr(obj, '__slots__'):
                slots_dict = {
                    slot: getattr(obj, slot)
                    for slot in obj.__slots__
                    if hasattr(obj, slot)
                }
                return (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    self._make_hashable(slots_dict, seen)
                )
            else:
                return str(obj)
        finally:
            if is_container:
                seen.remove(obj_id)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the memoized function or decorates a function.

        Args:
            *args (Any): The arguments to the function, or the function to decorate.
            **kwargs (Any): The keyword arguments to the function.

        Returns:
            Any: The result of the function call, or self if decorating.
        """
        if self.f is None:
            f = args[0]
            self.f = f
            return self

        try:
            serialized_args = self._make_hashable(args)
            serialized_kwargs = self._make_hashable(kwargs)
            cache_key = (serialized_args, serialized_kwargs)
        except Exception:
            return self.f(*args, **kwargs)

        if cache_key in self.cache:
            val = self.cache.pop(cache_key)
            self.cache[cache_key] = val
            return val

        result = self.f(*args, **kwargs)
        self.cache[cache_key] = result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return result

    def __get__(self, instance: Any, _owner: Any = None) -> Any:
        """Supports binding to active object instances.

        Ensures that instance methods are cached per-instance to prevent
        cross-instance cache leaks.

        Args:
            instance (Any): The instance to bind to.
            _owner (Any, optional): The owner class. Defaults to None.

        Returns:
            Any: A bound method wrapper that performs cached execution.
        """
        if instance is None:
            return self

        assert self.f is not None
        cache_attr = f"_memo_cache_{self.f.__name__}_{id(self)}"

        if not hasattr(instance, cache_attr):
            setattr(instance, cache_attr, OrderedDict())

        instance_cache = getattr(instance, cache_attr)

        def bound_method(*args: Any, **kwargs: Any) -> Any:
            """Executes the bound method with cached results.

            Args:
                *args (Any): The positional arguments.
                **kwargs (Any): The keyword arguments.

            Returns:
                Any: The result of the method execution.
            """
            assert self.f is not None
            try:
                serialized_args = self._make_hashable(args)
                serialized_kwargs = self._make_hashable(kwargs)
                cache_key = (serialized_args, serialized_kwargs)
            except Exception:
                return self.f(instance, *args, **kwargs)

            if cache_key in instance_cache:
                val = instance_cache.pop(cache_key)
                instance_cache[cache_key] = val
                return val

            result = self.f(instance, *args, **kwargs)
            instance_cache[cache_key] = result
            if len(instance_cache) > self.maxsize:
                instance_cache.popitem(last=False)
            return result

        return bound_method


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
    "iterable_to_json",
    "filter_kwargs_for_callable"
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

def filter_kwargs_for_callable(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filters kwargs so that only those accepted by func are returned, unless func accepts ``**kwargs``."""
    import inspect
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return kwargs

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        return kwargs

    filtered = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
    return filtered

