import json
from unittest.mock import mock_open, patch

from clintrials.utils import (
    fetch_json_from_files,
    filter_list_of_dicts,
    invoke_map_reduce_on_list,
    map_reduce_files,
    multiindex_dataframe_from_tuple_map,
    reduce_maps_by_summing,
    to_1d_list,
    levenshtein,
    levenshtein_index,
    support_match,
    Memoize,
)


def test_fetch_json_from_files():
    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = ["file1.json", "file2.json"]
        with patch("builtins.open", mock_open(read_data='[{"a": 1}]')) as mock_file:
            data = fetch_json_from_files("*.json")
            assert data == [{"a": 1}, {"a": 1}]
            assert mock_glob.call_count == 1
            assert mock_file.call_count == 2


def test_filter_list_of_dicts():
    list_of_dicts = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
        {"a": 2, "b": 2},
    ]
    assert filter_list_of_dicts(list_of_dicts, {"a": 1}) == [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    assert filter_list_of_dicts(list_of_dicts, {"b": 2}) == [
        {"a": 1, "b": 2},
        {"a": 2, "b": 2},
    ]
    assert filter_list_of_dicts(list_of_dicts, {"a": 1, "b": 2}) == [
        {"a": 1, "b": 2},
    ]


def test_map_reduce_files():
    with patch("builtins.open", mock_open(read_data='[{"a": 1}]')) as mock_file:
        data = map_reduce_files(
            ["file1.json", "file2.json"],
            lambda x: json.load(open(x)),
            lambda x, y: x + y,
        )
        assert data == [{"a": 1}, {"a": 1}]


def test_invoke_map_reduce_on_list():
    a_list = [1, 2, 3]
    function_map = {
        "sum": (lambda x: x, lambda x, y: x + y),
        "product": (lambda x: x, lambda x, y: x * y),
    }
    result = invoke_map_reduce_on_list(a_list, function_map)
    assert result == {"sum": 6, "product": 6}


def test_reduce_maps_by_summing():
    x = {"a": 1, "b": 2}
    y = {"a": 3, "b": 4}
    result = reduce_maps_by_summing(x, y)
    assert result == {"a": 4, "b": 6}


import pandas as pd


def test_multiindex_dataframe_from_tuple_map():
    x = {
        ("a", 1): 1,
        ("a", 2): 2,
        ("b", 1): 3,
        ("b", 2): 4,
    }
    labels = ["l1", "l2"]
    df = multiindex_dataframe_from_tuple_map(x, labels)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 1)

def test_to_1d_list():
    assert to_1d_list(1) == [1]
    assert to_1d_list([1, 2, 3]) == [1, 2, 3]
    assert to_1d_list([1, [2, 3]]) == [1, 2, 3]
    assert to_1d_list([1, [2, [3]]]) == [1, 2, 3]

def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("saturday", "sunday") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("abc", "abc") == 0

def test_levenshtein_index():
    assert levenshtein_index("kitten", "sitting") == 1 - 3/7
    assert levenshtein_index("saturday", "sunday") == 1 - 3/8
    assert levenshtein_index("", "abc") == 0.0
    assert levenshtein_index("abc", "abc") == 1.0

def test_support_match():
    assert support_match([1, 2, 3], [1, 2, 4]) == 4/6
    assert support_match([1, 2, 3], [4, 5, 6]) == 0
    assert support_match([1, 2, 3], [1, 2, 3]) == 1.0
    assert support_match([], []) == 0.0

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
