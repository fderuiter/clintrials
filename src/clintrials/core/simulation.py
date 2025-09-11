__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import itertools
import json
import logging
import warnings
from collections import OrderedDict
from datetime import datetime

from clintrials.utils import (
    _open_json_local,
    fetch_json_from_files,
    filter_list_of_dicts,
    invoke_map_reduce_on_list,
    map_reduce_files,
    multiindex_dataframe_from_tuple_map,
    reduce_maps_by_summing,
)

logger = logging.getLogger(__name__)


def run_sims(sim_func, n1=1, n2=1, out_file=None, **kwargs):
    """Runs simulations using a delegate function.

    Note:
        - A total of `n1 * n2` simulations are performed.
        - `sim_func` is expected to return a JSON-able object.
        - If `out_file` is provided, the results are saved after each batch.

    Args:
        sim_func: The delegate function to call for each simulation.
        n1: The number of batches.
        n2: The number of iterations per batch.
        out_file: The location of the file for incremental saving.
        **kwargs: Keyword arguments to pass to `sim_func`.

    Returns:
        A list of simulation results.
    """

    sims = []
    for j in range(n1):
        sims1 = [sim_func(**kwargs) for i in range(n2)]
        sims += sims1
        if out_file:
            try:
                with open(out_file, "w") as outfile:
                    json.dump(sims, outfile)
            except Exception as e:
                logger.error("Error writing: %s", e)
        logger.info(f"{j} {datetime.now()} {len(sims)}")
    return sims


def sim_parameter_space(sim_func, ps, n1=1, n2=None, out_file=None):
    """Runs simulations over a parameter space.

    Note:
        - A total of `n1 * n2` simulations are performed.
        - `sim_func` is expected to return a JSON-able object.
        - If `out_file` is provided, the results are saved after each batch.

    Args:
        sim_func: The function to call for each simulation. Parameters are
            provided via `ps` as unpacked keyword arguments.
        ps: The parameter space to explore.
        n1: The number of batches.
        n2: The number of iterations per batch. If not provided, it defaults
            to the size of the parameter space.
        out_file: The location of the file for incremental saving.

    Returns:
        A list of simulation results.
    """

    if not n2 or n2 <= 0:
        n2 = ps.size()

    sims = []
    params_iterator = ps.get_cyclical_iterator()
    for j in range(n1):
        sims1 = [sim_func(**params_iterator.next()) for i in range(n2)]
        sims += sims1
        if out_file:
            try:
                with open(out_file, "w") as outfile:
                    json.dump(sims, outfile)
            except Exception as e:
                logger.error("Error writing: %s", e)
        logger.info(f"{j} {datetime.now()} {len(sims)}")
    return sims


def go_fetch_json_sims(file_pattern):
    """Fetches JSON simulations from files.

    .. deprecated:: 0.1.4
        Use :func:`clintrials.utils.fetch_json_from_files` instead.
    """
    warnings.warn(
        "go_fetch_json_sims is deprecated; use fetch_json_from_files instead",
        DeprecationWarning,
    )
    return fetch_json_from_files(file_pattern)


def filter_sims(sims, filter_dict):
    """Filters a list of simulations.

    .. deprecated:: 0.1.4
        Use :func:`clintrials.utils.filter_list_of_dicts` instead.
    """
    warnings.warn(
        "filter_sims is deprecated; use filter_list_of_dicts instead",
        DeprecationWarning,
    )
    return filter_list_of_dicts(sims, filter_dict)


def extract_sim_data(sims, ps, func_map, var_map=None, return_type="dataframe"):
    """Extracts and summarizes a list of simulations.

    This method partitions simulations into subsets based on the parameter
    space, and then applies a collection of summary functions to each subset.

    Args:
        sims: A list of simulations (likely in JSON format).
        ps: The parameter space used to filter the simulations.
        func_map: A map of item names to functions that take a list of sims
            and a parameter map, and return a summary statistic.
        var_map: A map from variable names in the simulation JSON to argument
            names in the parameter space.
        return_type: The desired return type. Can be 'dataframe' (default)
            or 'tuple'.

    Returns:
        A pandas DataFrame or a tuple of lists containing the summarized data.
    """

    if var_map is None:
        var_names = ps.keys()
        var_map = {}
        for var_name in var_names:
            var_map[var_name] = var_name
    else:
        var_names = var_map.keys()

    z = [(var_name, ps[var_map[var_name]]) for var_name in var_names]
    labels, val_arrays = zip(*z)
    param_combinations = list(itertools.product(*val_arrays))
    index_tuples = []
    row_tuples = []
    for param_combo in param_combinations:
        these_params = dict(zip(labels, param_combo))
        these_sims = filter_list_of_dicts(sims, these_params)
        if len(these_sims):
            these_metrics = {
                label: func(these_sims, these_params)
                for label, func in func_map.items()
            }
            index_tuples.append(param_combo)
            row_tuples.append(these_metrics)
    if len(row_tuples):
        if return_type == "dataframe":
            import pandas as pd

            return pd.DataFrame(
                row_tuples, pd.MultiIndex.from_tuples(index_tuples, names=var_names)
            )
        else:
            return row_tuples, index_tuples
    else:
        if return_type == "dataframe":
            import pandas as pd

            return pd.DataFrame(columns=func_map.keys())
        else:
            return [], []


def summarise_sims(sims, ps, func_map, var_map=None, to_pandas=True):
    """Summarizes a list of simulations.

    .. deprecated:: 0.1.4
        Use :func:`extract_sim_data` instead.
    """
    import warnings

    warnings.warn(
        "summarise_sims is deprecated, use extract_sim_data instead",
        DeprecationWarning,
    )
    return_type = "dataframe" if to_pandas else "tuple"
    return extract_sim_data(sims, ps, func_map, var_map, return_type=return_type)


# Map-Reduce methods for summarising sims in memory-efficient ways
def invoke_map_reduce_function_map(sims, function_map):
    """Invokes a map-reduce pattern on a list of simulations.

    .. deprecated:: 0.1.4
        Use :func:`clintrials.utils.invoke_map_reduce_on_list` instead.
    """
    warnings.warn(
        "invoke_map_reduce_function_map is deprecated; use invoke_map_reduce_on_list instead",
        DeprecationWarning,
    )
    return invoke_map_reduce_on_list(sims, function_map)


# The following functions are helper functions for processing simulation results.
# They have been reviewed and are considered to be specific enough to the
# simulation workflow to remain in this module rather than being moved to a
# more general utility module.
def partition_and_aggregate(sims, ps, function_map):
    """Partitions and aggregates simulations.

    This function partitions simulations into subsets based on the parameter
    space, and then applies a collection of map-reduce function pairs to each
    subset.

    Args:
        sims: A list of simulations (likely in JSON format).
        ps: The parameter space used to filter the simulations.
        function_map: A map of item names to (map_func, reduce_func) pairs.

    Returns:
        A map of parameter combinations to the reduced object.
    """

    var_names = ps.keys()
    z = [(var_name, ps[var_name]) for var_name in var_names]
    labels, val_arrays = zip(*z)
    param_combinations = list(itertools.product(*val_arrays))
    out = OrderedDict()
    for param_combo in param_combinations:

        these_params = dict(zip(labels, param_combo))
        these_sims = filter_list_of_dicts(sims, these_params)

        out[param_combo] = invoke_map_reduce_on_list(these_sims, function_map)

    return out


def fetch_partition_and_aggregate(f, ps, function_map, verbose=False):
    """Fetches, partitions, and aggregates simulations from a file.

    This function loads JSON simulations from a file and then passes them
    to `partition_and_aggregate`.

    Args:
        f: The location of the file.
        ps: The parameter space used to filter the simulations.
        function_map: A map of item names to (map_func, reduce_func) pairs.
        verbose: If True, logs the number of simulations fetched.

    Returns:
        A map of parameter combinations to the reduced object.
    """

    sims = _open_json_local(f)
    if verbose:
        logger.info("Fetched %s sims from %s", len(sims), f)
    return partition_and_aggregate(sims, ps, function_map)


def reduce_product_of_two_files_by_summing(x, y):
    """Reduces the summaries of two files by summing their values.

    Args:
        x: The first summary dictionary.
        y: The second summary dictionary.

    Returns:
        A new dictionary with the summed values.
    """
    response = OrderedDict()
    for k in x.keys():
        response[k] = reduce_maps_by_summing(x[k], y[k])
    return response
