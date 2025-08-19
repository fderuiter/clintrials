import numpy as np

from clintrials.util import atomic_to_json, iterable_to_json


def test_atomic_to_json():
    assert atomic_to_json(1) == 1
    assert atomic_to_json(1.0) == 1.0
    assert atomic_to_json("a") == "a"
    assert atomic_to_json(True) is True
    assert atomic_to_json(np.int64(1)) == 1
    assert atomic_to_json(np.float64(1.0)) == 1.0


def test_iterable_to_json():
    assert iterable_to_json([1, 2, 3]) == [1, 2, 3]
    assert iterable_to_json((1, 2, 3)) == [1, 2, 3]
    assert iterable_to_json(np.array([1, 2, 3])) == [1, 2, 3]
