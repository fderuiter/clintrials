
from clintrials.utils import Memoize, filter_list_of_dicts, to_1d_list


def test_filter_list_of_dicts():
    list_of_dicts = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 2}]
    assert filter_list_of_dicts(list_of_dicts, {'a': 1}) == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
    assert filter_list_of_dicts(list_of_dicts, {'b': 2}) == [{'a': 1, 'b': 2}, {'a': 2, 'b': 2}]
    assert filter_list_of_dicts(list_of_dicts, {'a': 1, 'b': 2}) == [{'a': 1, 'b': 2}]

def test_to_1d_list():
    assert to_1d_list(1) == [1]
    assert to_1d_list([1, 2, 3]) == [1, 2, 3]
    assert to_1d_list([1, [2, 3]]) == [1, 2, 3]
    assert to_1d_list([1, [2, [3]]]) == [1, 2, 3]

def test_memoize():

    class MyClass:

        def __init__(self):
            self.call_count = 0

        @Memoize
        def my_method(self, x):
            self.call_count += 1
            return x * 2
    c = MyClass()
    assert c.my_method(2) == 4
    assert c.call_count == 1
    assert c.my_method(2) == 4
    assert c.call_count == 1
    assert c.my_method(3) == 6
    assert c.call_count == 2

import pytest

from clintrials.utils import deprecated


def test_deprecated_function():
    @deprecated(alternative="new_func")
    def old_func():
        return 42

    with pytest.warns(DeprecationWarning) as record:
        result = old_func()

    assert result == 42
    assert len(record) == 1
    assert "old_func is deprecated" in str(record[0].message)
    assert "Use new_func instead" in str(record[0].message)

def test_deprecated_class():
    @deprecated(alternative="NewClass")
    class OldClass:
        def __init__(self, val):
            self.val = val

    with pytest.warns(DeprecationWarning) as record:
        obj = OldClass(10)

    assert obj.val == 10
    assert len(record) == 1
    assert "OldClass is deprecated" in str(record[0].message)
    assert "Use NewClass instead" in str(record[0].message)
