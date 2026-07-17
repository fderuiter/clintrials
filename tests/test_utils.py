
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
