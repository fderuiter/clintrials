"""
Functions for running and analyzing clinical trial simulations.


Random Seed Strategy: {simulation_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import copy
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
    tuple_to_dataframe,
)

__all__ = [
    "run_sims",
    "sim_parameter_space",
    "extract_sim_data",
    "partition_and_aggregate",
    "fetch_partition_and_aggregate",
    "reduce_product_of_two_files_by_summing",
    "UniversalProtocolSimulationRunner"
]

logger = logging.getLogger(__name__)


def run_sims(
    sim_func, n1=1, n2=1, out_file=None, agg_func=None, metadata=None, **kwargs
):
    """Runs simulations using a delegate function.

    Args:
        sim_func (callable): The delegate function to be called to yield a
            single simulation.
        n1 (int, optional): The number of batches. Defaults to 1.
        n2 (int, optional): The number of iterations per batch. Defaults to 1.
        out_file (str, optional): The location of the file for incremental
            saving after each batch. Defaults to None.
        agg_func (callable, optional): Online aggregation function to process
            incremental results. Should have the signature `agg_func(current_sims, new_batch_sims)`.
        metadata (dict, optional): Self-describing metadata headers that detail
            simulation parameters alongside results.
        **kwargs: Keyword arguments to be passed to `sim_func`.

    Returns:
        list or dict: A list of simulation results, or if metadata is provided,
            a nested dict with "Parameters" and "Simulations" keys.
    """
    sims = [] if agg_func is None else None
    for j in range(n1):
        sims1 = [sim_func(**kwargs) for i in range(n2)]
        if agg_func:
            sims = agg_func(sims, sims1)
        else:
            sims += sims1
        if out_file:
            saved_to_idb = False
            try:
                import js
                import json
                batch_json = json.dumps(sims1)
                metadata_json = json.dumps(metadata) if metadata else "null"
                js_code = f"""
                (function() {{
                    try {{
                        var req = window.indexedDB.open('clintrials_db', 1);
                        req.onupgradeneeded = function(e) {{
                            var db = e.target.result;
                            if (!db.objectStoreNames.contains('simulations')) {{
                                var store = db.createObjectStore('simulations', {{ keyPath: 'id', autoIncrement: true }});
                                store.createIndex('out_file', 'out_file', {{ unique: false }});
                            }}
                        }};
                        req.onsuccess = function(e) {{
                            var db = e.target.result;
                            var tx = db.transaction('simulations', 'readwrite');
                            var store = tx.objectStore('simulations');
                            store.add({{
                                out_file: '{out_file}',
                                metadata: {metadata_json},
                                batch: JSON.parse({repr(batch_json)})
                            }});
                        }};
                    }} catch(err) {{
                        console.error("IDB save error", err);
                    }}
                }})();
                """
                js.eval(js_code)
                saved_to_idb = True
            except ImportError:
                pass
                
            if not saved_to_idb:
                try:
                    with open(out_file, "w") as outfile:
                        output = (
                            {"Parameters": metadata, "Simulations": sims}
                            if metadata is not None
                            else sims
                        )
                        json.dump(output, outfile)
                except Exception as e:
                    logger.error("Error writing: %s", e)
        sims_len = len(sims) if isinstance(sims, list) else "agg"
        logger.info(f"{j} {datetime.now()} {sims_len}")

    if metadata is not None:
        return {"Parameters": metadata, "Simulations": sims}
    return sims


def sim_parameter_space(
    sim_func, ps, n1=1, n2=None, out_file=None, agg_func=None, metadata=None
):
    """Runs simulations for a parameter space.

    Args:
        sim_func (callable): The function to be called to yield a single
            simulation. Parameters are provided via `ps` as unpacked kwargs.
        ps (clintrials.utils.ParameterSpace): The parameter space to explore.
        n1 (int, optional): The number of batches. Defaults to 1.
        n2 (int, optional): The number of iterations per batch. If None, it
            defaults to the size of the parameter space. Defaults to None.
        out_file (str, optional): The location of the file for incremental
            saving after each batch. Defaults to None.
        agg_func (callable, optional): Online aggregation function to process
            incremental results.
        metadata (dict, optional): Self-describing metadata headers that detail
            simulation parameters alongside results.

    Returns:
        list or dict: A list of simulation results, or if metadata is provided,
            a nested dict with "Parameters" and "Simulations" keys.
    """
    if not n2 or n2 <= 0:
        n2 = ps.size()

    sims = [] if agg_func is None else None
    params_iterator = ps.get_cyclical_iterator()
    for j in range(n1):
        sims1 = [sim_func(**params_iterator.next()) for i in range(n2)]
        if agg_func:
            sims = agg_func(sims, sims1)
        else:
            sims += sims1
        if out_file:
            saved_to_idb = False
            try:
                import js
                import json
                batch_json = json.dumps(sims1)
                metadata_json = json.dumps(metadata) if metadata else "null"
                js_code = f"""
                (function() {{
                    try {{
                        var req = window.indexedDB.open('clintrials_db', 1);
                        req.onupgradeneeded = function(e) {{
                            var db = e.target.result;
                            if (!db.objectStoreNames.contains('simulations')) {{
                                var store = db.createObjectStore('simulations', {{ keyPath: 'id', autoIncrement: true }});
                                store.createIndex('out_file', 'out_file', {{ unique: false }});
                            }}
                        }};
                        req.onsuccess = function(e) {{
                            var db = e.target.result;
                            var tx = db.transaction('simulations', 'readwrite');
                            var store = tx.objectStore('simulations');
                            store.add({{
                                out_file: '{out_file}',
                                metadata: {metadata_json},
                                batch: JSON.parse({repr(batch_json)})
                            }});
                        }};
                    }} catch(err) {{
                        console.error("IDB save error", err);
                    }}
                }})();
                """
                js.eval(js_code)
                saved_to_idb = True
            except ImportError:
                pass
                
            if not saved_to_idb:
                try:
                    with open(out_file, "w") as outfile:
                        output = (
                            {"Parameters": metadata, "Simulations": sims}
                            if metadata is not None
                            else sims
                        )
                        json.dump(output, outfile)
                except Exception as e:
                    logger.error("Error writing: %s", e)
        sims_len = len(sims) if isinstance(sims, list) else "agg"
        logger.info(f"{j} {datetime.now()} {sims_len}")

    if metadata is not None:
        return {"Parameters": metadata, "Simulations": sims}
    return sims






def extract_sim_data(sims, ps, func_map, var_map=None, return_type="dataframe"):
    """Extracts and summarises a list of simulations.

    This method partitions simulations into subsets that used the same set of
    parameters, and then invokes a collection of summary functions on each
    subset.

    Args:
        sims (list or pandas.DataFrame): A list of simulations, likely in JSON
            format, or a pandas DataFrame.
        ps (clintrials.utils.ParameterSpace): The parameter space that
            explains how to filter simulations.
        func_map (dict): A map from item name to a function that takes a list
            of sims and a parameter map as arguments and returns a summary
            statistic or object.
        var_map (dict, optional): A map from variable name in the simulation
            JSON to the argument name in the `ParameterSpace`. If None, it is
            assumed that the names are the same. Defaults to None.
        return_type (str, optional): The return type. 'dataframe' to get a
            pandas.DataFrame; 'tuple' to get a tuple of lists. Defaults to
            'dataframe'.

    Returns:
        pandas.DataFrame or tuple: A DataFrame with the summarised results,
            or a tuple of lists for backward compatibility.
    """
    # Normalize input
    try:
        import pandas as pd

        if isinstance(sims, pd.DataFrame):
            sims = sims.to_dict(orient="records")
    except ImportError:
        pass

    if var_map is None:
        var_names = list(ps.keys())
        var_map = {}
        for var_name in var_names:
            var_map[var_name] = var_name
    else:
        var_names = list(var_map.keys())

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

    if return_type == "dataframe":
        return tuple_to_dataframe(
            row_tuples,
            index_tuples,
            column_names=func_map.keys(),
            index_names=var_names,
        )
    else:
        return row_tuples, index_tuples




# Map-Reduce methods for summarising sims in memory-efficient ways


# The following functions are helper functions for processing simulation results.
# They have been reviewed and are considered to be specific enough to the
# simulation workflow to remain in this module rather than being moved to a
# more general utility module.
def partition_and_aggregate(sims, ps, function_map):
    """Partitions and aggregates simulations.

    This function partitions simulations into subsets that used the same set
    of parameters, and then invokes a collection of map/reduce function pairs
    on each subset.

    Args:
        sims (list): A list of simulations, likely in JSON format.
        ps (clintrials.utils.ParameterSpace): The parameter space that
            explains how to filter simulations.
        function_map (dict): A map of item to (map_func, reduce_func) pairs.

    Returns:
        dict: A map of parameter combination to the reduced object.
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

    This function loads JSON simulations from a file and then passes them to
    `partition_and_aggregate`.

    Args:
        f (str): The file location.
        ps (clintrials.utils.ParameterSpace): The parameter space that
            explains how to filter simulations.
        function_map (dict): A map of item to (map_func, reduce_func) pairs.
        verbose (bool, optional): If `True`, logs the number of sims fetched.
            Defaults to `False`.

    Returns:
        dict: A map of parameter combination to the reduced object.
    """
    sims = _open_json_local(f)
    if verbose:
        logger.info("Fetched %s sims from %s", len(sims), f)
    return partition_and_aggregate(sims, ps, function_map)


def reduce_product_of_two_files_by_summing(x, y):
    """Reduces the summaries of two files by summing their values.

    Args:
        x (dict): The first summary dictionary.
        y (dict): The second summary dictionary.

    Returns:
        dict: A new dictionary with the summed values.
    """
    response = OrderedDict()
    for k in x.keys():
        response[k] = reduce_maps_by_summing(x[k], y[k])
    return response

class UniversalProtocolSimulationRunner:
    """Universal Protocol Simulation Runner for executing trial designs.
    
    This runner standardises the simulation loop across all trial types, handling
    initialisation, recruitment timing, outcome generation, and standard reporting.
    """

    def __init__(self, design, outcome_generator=None, recruitment_stream=None):
        self.design = design
        self.outcome_generator = outcome_generator
        self.recruitment_stream = recruitment_stream

    def run(self, cohort_size=1, **kwargs):
        """Runs the trial simulation loop.
        
        Args:
            cohort_size (int): Number of patients per cohort. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the outcome
                generator and the design's update method.
        
        Returns:
            collections.OrderedDict: The trial simulation report.
        """
        design = copy.deepcopy(self.design)
        recruitment_stream = copy.deepcopy(self.recruitment_stream) if self.recruitment_stream else None

        design.reset()
        if recruitment_stream:
            recruitment_stream.reset()

        i = 0
        max_size = design.max_size()

        while i < max_size and design.has_more():
            current_cohort_size = min(cohort_size, max_size - i)

            if recruitment_stream:
                kwargs["arrival_times"] = [
                    recruitment_stream.next() for _ in range(current_cohort_size)
                ]

            if self.outcome_generator:
                cases = self.outcome_generator(
                    design=design,
                    current_size=i,
                    cohort_size=current_cohort_size,
                    **kwargs
                )
            else:
                cases = []

            design.update(cases, **kwargs)
            i += current_cohort_size

        return design.report()



# Legacy imports for backward compatibility
from clintrials.legacy.simulation import (
    go_fetch_json_sims,
    filter_sims,
    summarise_sims,
    invoke_map_reduce_function_map,
)

# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY
    __doc__ = __doc__.format(**REGISTRY)
