# SPDX-License-Identifier: MIT


from clintrials._utils import filter_list_of_dicts, to_1d_list


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
