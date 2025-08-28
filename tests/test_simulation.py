import pandas as pd
import pytest

from clintrials.core.simulation import extract_sim_data, summarise_sims
from clintrials.utils import ParameterSpace


@pytest.fixture
def sample_sims():
    """Return a list of sample simulation results."""
    return [
        {"param1": 1, "param2": "a", "metric": 10},
        {"param1": 1, "param2": "b", "metric": 20},
        {"param1": 2, "param2": "a", "metric": 30},
        {"param1": 2, "param2": "b", "metric": 40},
        {"param1": 1, "param2": "a", "metric": 15},
    ]


@pytest.fixture
def sample_ps():
    """Return a sample ParameterSpace."""
    ps = ParameterSpace()
    ps.add("param1", [1, 2])
    ps.add("param2", ["a", "b"])
    return ps


def mean_metric(sims, params):
    """Calculate the mean of the 'metric' field."""
    return pd.DataFrame(sims)["metric"].mean()


def test_extract_sim_data_dataframe_default(sample_sims, sample_ps):
    """Test that extract_sim_data returns a DataFrame by default."""
    func_map = {"mean_metric": mean_metric}
    result = extract_sim_data(sample_sims, sample_ps, func_map)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "mean_metric" in result.columns
    assert result.loc[(1, "a"), "mean_metric"] == 12.5
    assert result.loc[(1, "b"), "mean_metric"] == 20
    assert result.loc[(2, "a"), "mean_metric"] == 30
    assert result.loc[(2, "b"), "mean_metric"] == 40


def test_extract_sim_data_tuple(sample_sims, sample_ps):
    """Test that extract_sim_data returns a tuple when requested."""
    func_map = {"mean_metric": mean_metric}
    result = extract_sim_data(sample_sims, sample_ps, func_map, return_type="tuple")
    assert isinstance(result, tuple)
    assert len(result) == 2
    row_tuples, index_tuples = result
    assert isinstance(row_tuples, list)
    assert isinstance(index_tuples, list)
    assert len(row_tuples) == 4
    assert len(index_tuples) == 4


def test_extract_sim_data_empty_sims(sample_ps):
    """Test that extract_sim_data handles empty simulation lists correctly."""
    func_map = {"mean_metric": mean_metric}

    # Test DataFrame return type
    result_df = extract_sim_data([], sample_ps, func_map)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    assert "mean_metric" in result_df.columns

    # Test tuple return type
    result_tuple = extract_sim_data([], sample_ps, func_map, return_type="tuple")
    assert result_tuple == ([], [])


def test_summarise_sims_deprecated(sample_sims, sample_ps):
    """Test that the old summarise_sims function is deprecated but functional."""
    func_map = {"mean_metric": mean_metric}
    with pytest.deprecated_call():
        result = summarise_sims(sample_sims, sample_ps, func_map, to_pandas=True)
        assert isinstance(result, pd.DataFrame)

    with pytest.deprecated_call():
        result_tuple = summarise_sims(sample_sims, sample_ps, func_map, to_pandas=False)
        assert isinstance(result_tuple, tuple)
