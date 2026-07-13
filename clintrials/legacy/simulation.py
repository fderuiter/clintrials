import warnings
from clintrials.utils import (
    fetch_json_from_files,
    filter_list_of_dicts,
    invoke_map_reduce_on_list,
)

def go_fetch_json_sims(file_pattern):
    """Fetches and combines JSON data from multiple files.

    .. deprecated:: 0.1.4
       Use `fetch_json_from_files` instead.
    """
    warnings.warn(
        "go_fetch_json_sims is deprecated; use fetch_json_from_files instead",
        DeprecationWarning,
    )
    return fetch_json_from_files(file_pattern)

def filter_sims(sims, filter_dict):
    """Filters a list of dictionaries based on a filter dictionary.

    .. deprecated:: 0.1.4
       Use `filter_list_of_dicts` instead.
    """
    warnings.warn(
        "filter_sims is deprecated; use filter_list_of_dicts instead",
        DeprecationWarning,
    )
    return filter_list_of_dicts(sims, filter_dict)

def summarise_sims(sims, ps, func_map, var_map=None, to_pandas=True):
    """Summarises a list of simulations.

    .. deprecated:: 0.1.4
       Use `extract_sim_data` instead.
    """
    warnings.warn(
        "summarise_sims is deprecated; use extract_sim_data instead",
        DeprecationWarning,
    )
    from clintrials.core.simulation import extract_sim_data
    return_type = "dataframe" if to_pandas else "tuple"
    return extract_sim_data(sims, ps, func_map, var_map, return_type=return_type)

def invoke_map_reduce_function_map(sims, function_map):
    """Invokes a map-reduce pattern on a list.

    .. deprecated:: 0.1.4
       Use `invoke_map_reduce_on_list` instead.
    """
    warnings.warn(
        "invoke_map_reduce_function_map is deprecated; use invoke_map_reduce_on_list instead",
        DeprecationWarning,
    )
    return invoke_map_reduce_on_list(sims, function_map)

