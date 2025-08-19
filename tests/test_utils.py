import json
from unittest.mock import mock_open, patch

from clintrials.utils import (
    fetch_json_from_files,
    filter_list_of_dicts,
    invoke_map_reduce_on_list,
    map_reduce_files,
    multiindex_dataframe_from_tuple_map,
    reduce_maps_by_summing,
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
