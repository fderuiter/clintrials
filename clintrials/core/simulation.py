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
    """Run simulations using a delegate function.

    :param sim_func: Delegate function to be called to yield single simulation.
    :type sim_func: func
    :param n1: Number of batches
    :type n1: int
    :param n2: Number of iterations per batch
    :type n2: int
    :param out_file: Location of file for incremental saving after completion of each batch.
    :type out_file: str
    :param kwargs: key-word args for sim_func
    :type kwargs: dict

    .. note::

        - n1 * n2 simualtions are performed, in all.
        - sim_func is expected to return a JSON-able object
        - file is saved after each of n1 iterations, where applicable.

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
    """Run simulations using a function and a ParameterSpace.

    :param sim_func: function to be called to yield single simulation. Parameters are provided via ps as unpacked kwargs
    :type sim_func: func
    :param ps: Parameter space to explore via simulation
    :type ps: clintrials.utils.ParameterSpace
    :param n1: Number of batches
    :type n1: int
    :param n2: Number of iterations per batch
    :type n2: int
    :param out_file: Location of file for incremental saving after completion of each batch.
    :type out_file: str

    .. note::

        - n1 * n2 simualtions are performed, in all.
        - sim_func is expected to return a JSON-able object
        - file is saved after each of n1 iterations, where applicable.

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
    warnings.warn(
        "go_fetch_json_sims is deprecated; use fetch_json_from_files instead",
        DeprecationWarning,
    )
    return fetch_json_from_files(file_pattern)


def filter_sims(sims, filter_dict):
    warnings.warn(
        "filter_sims is deprecated; use filter_list_of_dicts instead",
        DeprecationWarning,
    )
    return filter_list_of_dicts(sims, filter_dict)


def extract_sim_data(sims, ps, func_map, var_map=None, return_type="dataframe"):
    """Extract and summarise a list of simulations.

    Method partitions simulations into subsets that used the same set of parameters, and then invokes
    a collection of summary functions on each subset.

    The return type is a pandas DataFrame by default, but can be switched to a tuple of lists
    for backward compatibility.

    :param sims: list of simulations (probably in JSON format)
    :type sims: list
    :param ps: ParameterSpace that will explain how to filter simulations
    :type ps: ParameterSpace
    :param var_map: map from variable name in simulation JSON to arg name in ParameterSpace
    :type var_map: dict
    :param func_map: map from item name to function that takes list of sims and parameter map as args and returns
                        a summary statistic or object.
    :type func_map: dict
    :param return_type: 'dataframe' to get a pandas.DataFrame; 'tuple' to get a tuple of lists.
    :type return_type: str

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
    """Summarise a list of simulations. (DEPRECATED)"""
    import warnings

    warnings.warn(
        "summarise_sims is deprecated, use extract_sim_data instead",
        DeprecationWarning,
    )
    return_type = "dataframe" if to_pandas else "tuple"
    return extract_sim_data(sims, ps, func_map, var_map, return_type=return_type)


# Map-Reduce methods for summarising sims in memory-efficient ways
def invoke_map_reduce_function_map(sims, function_map):
    warnings.warn(
        "invoke_map_reduce_function_map is deprecated; use invoke_map_reduce_on_list instead",
        DeprecationWarning,
    )
    return invoke_map_reduce_on_list(sims, function_map)


# I wrote the functions below during a specific analysis.
def partition_and_aggregate(sims, ps, function_map):
    """Function partitions simulations into subsets that used the same set of parameters,
    and then invokes a collection of map/reduce function pairs on each subset.

    :param sims: list of simulations (probably in JSON format)
    :type sims: list
    :param ps: ParameterSpace that will explain how to filter simulations
    :type ps: ParameterSpace
    :param function_map: map of item -> (map_func, reduce_func) pairs
    :type function_map: dict

    :returns: map of parameter combination to reduced object
    :rtype: dict

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
    """Function loads JSON sims in file f and then hands off to partition_and_aggregate.

    :param f: file location
    :type f: str
    :param ps: ParameterSpace that will explain how to filter simulations
    :type ps: ParameterSpace
    :param function_map: map of item -> (map_func, reduce_func) pairs
    :type function_map: dict

    :returns: map of parameter combination to reduced object
    :rtype: dict

    """

    sims = _open_json_local(f)
    if verbose:
        logger.info("Fetched %s sims from %s", len(sims), f)
    return partition_and_aggregate(sims, ps, function_map)


def reduce_product_of_two_files_by_summing(x, y):
    """Reduce the summaries of two files by summing."""
    response = OrderedDict()
    for k in x.keys():
        response[k] = reduce_maps_by_summing(x[k], y[k])
    return response
