# SPDX-License-Identifier: MIT

import numpy as np

from clintrials._utils import (
    atomic_to_json,
    filter_kwargs_for_callable,
    filter_list_of_dicts,
    iterable_to_json,
    to_1d_list,
    to_1d_list_gen,
)


def test_filter_list_of_dicts():
    list_of_dicts = [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 2}]
    assert filter_list_of_dicts(list_of_dicts, {"a": 1}) == [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    assert filter_list_of_dicts(list_of_dicts, {"b": 2}) == [
        {"a": 1, "b": 2},
        {"a": 2, "b": 2},
    ]
    assert filter_list_of_dicts(list_of_dicts, {"a": 1, "b": 2}) == [{"a": 1, "b": 2}]
    # Test filtering with tuple values
    assert filter_list_of_dicts([{"x": (1, 2)}], {"x": (1, 2)}) == [{"x": (1, 2)}]


def test_to_1d_list():
    assert to_1d_list(1) == [1]
    assert to_1d_list([1, 2, 3]) == [1, 2, 3]
    assert to_1d_list([1, [2, 3]]) == [1, 2, 3]
    assert to_1d_list([1, [2, [3]]]) == [1, 2, 3]


def test_to_1d_list_gen():
    assert list(to_1d_list_gen(1)) == [1]
    assert list(to_1d_list_gen([1, [2, [3]]])) == [1, 2, 3]


def test_atomic_to_json():
    # Test numpy generic
    val = np.int64(42)
    res = atomic_to_json(val)
    assert isinstance(res, int)
    assert res == 42

    # Test standard python type
    assert atomic_to_json("hello") == "hello"


def test_iterable_to_json():
    # Test iterable converting numpy objects
    vals = [np.int64(1), np.float64(2.5)]
    res = iterable_to_json(vals)
    assert res == [1, 2.5]
    assert all(isinstance(x, (int, float)) for x in res)

    # Test non-iterable
    assert iterable_to_json(np.int64(42)) == 42


def test_filter_kwargs_for_callable():
    def sample_func(a, b, c=1):
        return a + b + c

    kwargs = {"a": 1, "b": 2, "d": 4}
    res = filter_kwargs_for_callable(sample_func, kwargs)
    assert res == {"a": 1, "b": 2}

    # Test function with **kwargs
    def sample_with_var_kwargs(a, **kwargs):
        pass

    kwargs2 = {"a": 1, "extra": 42}
    assert filter_kwargs_for_callable(sample_with_var_kwargs, kwargs2) == kwargs2
