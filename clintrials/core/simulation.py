"""
Functions for running and analyzing clinical trial simulations.


Random Seed Strategy: {simulation_seed_strategy}
"""
__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'
import copy
import itertools
import json
import logging
from collections import OrderedDict
from datetime import datetime
from clintrials.utils import filter_list_of_dicts, tuple_to_dataframe

__all__ = [
    "run_sims",
    "sim_parameter_space",
    "extract_sim_data",
    "UniversalProtocolSimulationRunner"
]

logger = logging.getLogger(__name__)

def run_sims(sim_func, n1=1, n2=1, out_file=None, agg_func=None, metadata=None, **kwargs):
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
            try:
                with open(out_file, 'w') as outfile:
                    output = {'Parameters': metadata, 'Simulations': sims} if metadata is not None else sims
                    json.dump(output, outfile)
            except Exception as e:
                logger.error('Error writing: %s', e)
        sims_len = len(sims) if isinstance(sims, list) else 'agg'
        logger.info(f'{j} {datetime.now()} {sims_len}')
    if metadata is not None:
        return {'Parameters': metadata, 'Simulations': sims}
    return sims

def sim_parameter_space(sim_func, ps, n1=1, n2=None, out_file=None, agg_func=None, metadata=None):
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
            try:
                with open(out_file, 'w') as outfile:
                    output = {'Parameters': metadata, 'Simulations': sims} if metadata is not None else sims
                    json.dump(output, outfile)
            except Exception as e:
                logger.error('Error writing: %s', e)
        sims_len = len(sims) if isinstance(sims, list) else 'agg'
        logger.info(f'{j} {datetime.now()} {sims_len}')
    if metadata is not None:
        return {'Parameters': metadata, 'Simulations': sims}
    return sims

def extract_sim_data(sims, ps, func_map, var_map=None, return_type='dataframe'):
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
    try:
        import pandas as pd
        if isinstance(sims, pd.DataFrame):
            sims = sims.to_dict(orient='records')
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
            these_metrics = {label: func(these_sims, these_params) for label, func in func_map.items()}
            index_tuples.append(param_combo)
            row_tuples.append(these_metrics)
    if return_type == 'dataframe':
        return tuple_to_dataframe(row_tuples, index_tuples, column_names=func_map.keys(), index_names=var_names)
    else:
        return (row_tuples, index_tuples)

class UniversalProtocolSimulationRunner:
    """Universal Protocol Simulation Runner for executing trial designs.
    
    This runner standardises the simulation loop across all trial types, handling
    initialisation, recruitment timing, outcome generation, and standard reporting.
    """

    def __init__(self, design, outcome_generator=None, recruitment_stream=None):
        self.design = design
        self.outcome_generator = outcome_generator
        self.recruitment_stream = recruitment_stream

    def run(self, mode='iterative', n_sims=1, cohort_size=1, show_progress=False, **kwargs):
        """Runs the trial simulation loop.
        
        Args:
            mode (str): Execution mode, 'iterative' or 'vectorized'.
            n_sims (int): Number of simulations to run. Defaults to 1.
            cohort_size (int): Number of patients per cohort. Defaults to 1.
            show_progress (bool): Whether to show progress tracking.
            **kwargs: Additional keyword arguments passed to the outcome
                generator and the design's update method.
        
        Returns:
            list: A list of individual trial simulation reports.
        """
        if mode == 'vectorized':
            return self._run_vectorized(n_sims, **kwargs)
        try:
            from tqdm import tqdm
        except ImportError:
            show_progress = False
        results = []
        iterator = range(n_sims)
        if show_progress:
            iterator = tqdm(iterator, desc='Iterative Simulation')
        for _ in iterator:
            design = copy.deepcopy(self.design)
            recruitment_stream = copy.deepcopy(self.recruitment_stream) if self.recruitment_stream else None
            design.reset()
            if recruitment_stream:
                recruitment_stream.reset()
            i = 0
            max_size = getattr(design, 'max_size', lambda: float('inf'))()
            while i < max_size and design.has_more():
                current_cohort_size = min(cohort_size, max_size - i) if max_size != float('inf') else cohort_size
                if recruitment_stream:
                    kwargs['arrival_times'] = [recruitment_stream.next() for _ in range(current_cohort_size)]
                if self.outcome_generator:
                    cases = self.outcome_generator(design=design, current_size=i, cohort_size=current_cohort_size, **kwargs)
                else:
                    cases = []
                if self.outcome_generator:
                    design.update(cases, **kwargs)
                else:
                    design.update(**kwargs)
                i += current_cohort_size
            results.append(design.report())
        return results if n_sims > 1 or mode == 'iterative' else results[0]

    def _run_vectorized(self, n_sims, **kwargs):
        import numpy as np
        from clintrials.validation import validate_positive_integer
        validate_positive_integer(n_sims, 'Number of simulations')
        if hasattr(self.design, 'efficacy_boundaries'):
            theta = kwargs.get('theta', 0.0)
            means = theta * np.array(self.design.timing)
            cov = np.identity(self.design.k)
            for i in range(self.design.k):
                for j in range(i + 1, self.design.k):
                    corr = np.sqrt(self.design.timing[i] / self.design.timing[j])
                    cov[i, j] = cov[j, i] = corr
            simulated_z = self.design.rng.multivariate_normal(mean=means, cov=cov, size=n_sims)
            stopped_at = np.full(n_sims, self.design.k + 1, dtype=int)
            rejected = np.zeros(n_sims, dtype=bool)
            for i in range(self.design.k):
                ongoing_trials = stopped_at == self.design.k + 1
                stopping_now = simulated_z[:, i] >= self.design.efficacy_boundaries[i]
                update_mask = ongoing_trials & stopping_now
                stopped_at[update_mask] = i + 1
                rejected[update_mask] = True
            results = []
            for s, r, z in zip(stopped_at, rejected, simulated_z):
                s = int(s)
                actual_s = min(s, self.design.k)
                actual_z = z[:actual_s].tolist()
                actual_info = list(self.design.timing[:actual_s])
                report = OrderedDict()
                report['Stage'] = actual_s
                report['Stopped'] = True
                report['Rejected'] = bool(r)
                report['ZScores'] = actual_z
                report['Information'] = actual_info
                results.append(report)
            return results
        elif hasattr(self.design, 'run_bulk'):
            return self.design.run_bulk(n_sims, **kwargs)
        else:
            raise NotImplementedError('Vectorized mode not supported for this protocol type.')
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)