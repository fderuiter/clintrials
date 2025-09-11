__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

import glob
import json
import logging
from collections import OrderedDict
from collections.abc import Iterable
from copy import copy
from datetime import datetime
from functools import reduce
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)


def to_1d_list_gen(x):
    """Generator function to flatten a list of lists.

    This function recursively flattens a list of lists of arbitrary depth
    to a single list.

    Yields:
        object: The next item in the flattened list.
    """
    if isinstance(x, list):
        for y in x:
            yield from to_1d_list_gen(y)
    else:
        yield x


def to_1d_list(x):
    """Flattens a list of lists of arbitrary depth to a single list.

    Args:
        x (list or object): The list or object to flatten.

    Returns:
        list: A flattened list.
    """
    return list(to_1d_list_gen(x))


def get_logger(name: str = __name__) -> logging.Logger:
    """Gets a logger instance.

    Args:
        name (str, optional): The name of the logger. Defaults to `__name__`.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)


def _open_json_local(file_loc):
    """Opens a local JSON file.

    Args:
        file_loc (str): The path to the file.

    Returns:
        dict: The loaded JSON data.
    """
    return json.load(open(file_loc))


def _open_json_url(url):
    """Opens a JSON file from a URL.

    Args:
        url (str): The URL of the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    from urllib.request import urlopen

    return json.load(urlopen(url))


def fetch_json_from_files(file_pattern):
    """Fetches and combines JSON data from multiple files.

    Args:
        file_pattern (str): A glob pattern for the input files.

    Returns:
        list: A list of combined JSON objects.
    """
    files = glob.glob(file_pattern)
    sims = []
    for f in files:
        sub_sims = _open_json_local(f)
        logger.info("%s %s", f, len(sub_sims))
        sims += sub_sims
    logger.info("Fetched %s sims", len(sims))
    return sims


def filter_list_of_dicts(list_of_dicts, filter_dict):
    """Filters a list of dictionaries based on a filter dictionary.

    Args:
        list_of_dicts (list[dict]): The list of dictionaries to filter.
        filter_dict (dict): A dictionary of key-value pairs to filter by.

    Returns:
        list[dict]: The filtered list of dictionaries.
    """
    for key, val in filter_dict.items():
        if isinstance(val, (tuple)):
            list_of_dicts = [
                x for x in list_of_dicts if x[key] == val or x[key] == list(val)
            ]
        else:
            list_of_dicts = [x for x in list_of_dicts if x[key] == val]
    return list_of_dicts


def map_reduce_files(files, map_func, reduce_func):
    """Applies a map-reduce pattern to a list of files.

    Args:
        files (list[str]): A list of file paths.
        map_func (callable): The map function to apply to each file.
        reduce_func (callable): The reduce function to combine the results.

    Returns:
        object: The final reduced result.
    """
    if len(files):
        x = map(map_func, files)
        return reduce(reduce_func, x)
    else:
        raise TypeError("No files")


def invoke_map_reduce_on_list(a_list, function_map):
    """Invokes a map-reduce pattern on a list.

    Args:
        a_list (list): The list to process.
        function_map (dict): A dictionary mapping item names to
            (map_func, reduce_func) pairs.

    Returns:
        collections.OrderedDict: A dictionary of the reduced results.
    """
    response = OrderedDict()
    for item, function_tuple in function_map.items():
        map_func, reduce_func = function_tuple
        x = reduce(reduce_func, map(map_func, a_list))
        response[item] = x

    return response


def reduce_maps_by_summing(x, y):
    """Reduces two maps by summing their values.

    Args:
        x (dict): The first map.
        y (dict): The second map.

    Returns:
        collections.OrderedDict: A new map with the summed values.
    """
    response = OrderedDict()
    for k in x.keys():
        response[k] = x[k] + y[k]
    return response


def multiindex_dataframe_from_tuple_map(x, labels):
    """Creates a pandas DataFrame with a multi-index from a map.

    Args:
        x (dict): A map of (parameter-tuple -> value) pairs.
        labels (list[str]): A list of labels for the index levels.

    Returns:
        pandas.DataFrame: The resulting DataFrame.
    """
    import pandas as pd

    k, v = zip(*[(k, v) for (k, v) in x.items()])
    i = pd.MultiIndex.from_tuples(k, names=labels)
    return pd.DataFrame(list(v), index=i)


def fullname(o):
    """Gets the fully-qualified class name of an object.

    Args:
        o (object): The object.

    Returns:
        str: The fully-qualified class name.
    """
    return o.__module__ + "." + o.__class__.__name__


def atomic_to_json(obj):
    """Converts an atomic object to a JSON-friendly format.

    Args:
        obj (object): The object to convert.

    Returns:
        object: The JSON-friendly object.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def iterable_to_json(obj):
    """Converts an iterable to a JSON-friendly list.

    Args:
        obj (iterable): The iterable to convert.

    Returns:
        list: A list of JSON-friendly objects.
    """
    if isinstance(obj, Iterable):
        return [atomic_to_json(x) for x in obj]
    else:
        return atomic_to_json(obj)


def row_to_json(row, **kwargs):
    """Converts a pandas Series to a JSON-friendly dictionary.

    Args:
        row (pandas.Series): The Series to convert.
        **kwargs: Keyword arguments for `json.loads`.

    Returns:
        dict: The JSON-friendly dictionary.
    """
    try:
        doc = json.loads(row.to_json(), **kwargs)
    except UnicodeDecodeError:
        return row_to_json(row, encoding="iso-8859-1")
    import pandas as pd

    for x in row.index:
        if isinstance(row[x], datetime) and not pd.isnull(row[x]):
            doc[x] = pd.to_datetime(row[x]).date().isoformat()
    return doc


def _serialize_table_structure(df):
    """Serializes the structure of a DataFrame to a dictionary."""
    doc = OrderedDict()
    doc["Format"] = "Table"
    rows = []
    for i, row_name in enumerate(df.index):
        rows.append(
            OrderedDict(
                [
                    ("ID", str(i)),
                    ("Position", i + 1),
                    ("Label", atomic_to_json(row_name)),
                ]
            )
        )
    doc["Rows"] = rows
    doc["NumRows"] = len(df.index)
    cols = []
    for i, col_name in enumerate(df):
        cols.append(
            OrderedDict(
                [
                    ("ID", str(i)),
                    ("Position", i + 1),
                    ("Label", atomic_to_json(col_name)),
                ]
            )
        )
    doc["Cols"] = cols
    doc["NumCols"] = len(df.columns)
    table_data = OrderedDict()
    for j, col_name in enumerate(df):
        col_data = OrderedDict()
        for i, o in enumerate(df[col_name]):
            col_data[i] = atomic_to_json(o)
        table_data[str(j)] = col_data
    doc["Data"] = table_data
    return doc


def _calculate_value_counts(df, definitely_do_value_counts=False):
    """Calculates value counts for each column in a DataFrame."""
    freqs = OrderedDict()
    for col_name in df:
        vc = df[col_name].value_counts()
        if len(vc) < len(df) or definitely_do_value_counts:
            freqs[atomic_to_json(col_name)] = {
                atomic_to_json(k): atomic_to_json(v) for k, v in vc.items()
            }
    return freqs


def _calculate_column_summaries(df):
    """Calculates summary statistics for each column in a DataFrame."""
    col_summaries = OrderedDict()
    for i, col_name in enumerate(df):
        col_summary = OrderedDict()
        try:
            col_summary["Mean"] = df[col_name].mean()
        except:
            pass
        try:
            col_summary["Sum"] = df[col_name].sum()
        except:
            pass
        col_summaries[str(i)] = col_summary
    return col_summaries


def _calculate_row_summaries(df):
    """Calculates summary statistics for each row in a DataFrame."""
    row_summaries = OrderedDict()
    for i, row_name in enumerate(df.index):
        row_summary = OrderedDict()
        try:
            row_summary["Sum"] = df.loc[row_name].sum()
        except:
            pass
        row_summaries[str(i)] = row_summary
    return row_summaries


def df_to_json(
    df,
    do_value_counts=True,
    definitely_do_value_counts=False,
    do_column_summaries=True,
    do_row_summaries=True,
):
    """Serializes a pandas DataFrame to a JSON-friendly object.

    Args:
        df (pandas.DataFrame): The DataFrame to serialize.
        do_value_counts (bool, optional): If `True`, calculates value counts
            for each column. Defaults to `True`.
        definitely_do_value_counts (bool, optional): If `True`, forces the
            calculation of value counts even if all elements are unique.
            Defaults to `False`.
        do_column_summaries (bool, optional): If `True`, calculates summary
            statistics for each column. Defaults to `True`.
        do_row_summaries (bool, optional): If `True`, calculates summary
            statistics for each row. Defaults to `True`.

    Returns:
        dict: A JSON-friendly representation of the DataFrame.
    """
    doc = _serialize_table_structure(df)

    if do_value_counts:
        doc["Frequencies"] = _calculate_value_counts(df, definitely_do_value_counts)

    if do_column_summaries:
        doc["ColumnSummary"] = _calculate_column_summaries(df)

    if do_row_summaries:
        doc["RowSummary"] = _calculate_row_summaries(df)

    return doc


def levenshtein(s1, s2):
    """Calculates the Levenshtein distance between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The Levenshtein distance.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_index(s1, s2):
    """Calculates a similarity score between two strings based on the
    Levenshtein distance.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        float: A similarity score between 0 and 1.
    """
    l = levenshtein(s1, s2)
    max_length = max(len(s1), len(s2))
    if max_length:
        return 1 - 1.0 * l / max_length
    else:
        return 0.0


def support_match(a, b):
    """Calculates a support match score between two collections.

    The score is the percentage of elements of `a` in `b` and `b` in `a`.

    Args:
        a (iterable): The first collection.
        b (iterable): The second collection.

    Returns:
        float: A match score between 0.0 and 1.0.
    """
    try:
        a_set = set(a)
        b_set = set(b)
        a_in_b = [x in b_set for x in a_set]
        b_in_a = [x in a_set for x in b_set]
        return 1.0 * (sum(a_in_b) + sum(b_in_a)) / (len(a_set) + len(b_set))
    except:
        return 0.0


def _correlated_binary_outcomes_mardia(a, b, c):
    """Helper function for `correlated_binary_outcomes`."""
    if a == 0:
        return -c / b

    if b > 0:
        k = 1
    elif b < 0:
        k = -1
    else:
        k = 0
    p = -0.5 * (b + k * np.sqrt(b**2 - 4 * a * c))
    r1 = 1.0 * p / a
    r2 = 1.0 * c / p
    r = r2 if r2 > 0 else r1
    return r


def _correlated_binary_outcomes_solve2(mui, muj, psi):
    """Helper function for `correlated_binary_outcomes`."""
    if psi == 1:
        return mui * muj
    else:
        a = 1 - psi
        b = 1 - a * (mui + muj)
        c = -psi * (mui * muj)
        muij = _correlated_binary_outcomes_mardia(a, b, c)
    return muij


def correlated_binary_outcomes(num_pairs, u, psi, seed=None):
    """Generates correlated binary outcomes.

    This function uses the method from the R-package `ranBin2`.

    Args:
        num_pairs (int): The number of pairs to generate.
        u (list or tuple): A 2-item list or tuple of event probabilities.
        psi (float): The odds ratio of the binary outcomes.
        seed (int, optional): A seed for the random number generator.
            Defaults to `None`.

    Returns:
        numpy.ndarray: A 2D array of paired binary outcomes.
    """
    if seed:
        np.random.seed(seed)

    u12 = _correlated_binary_outcomes_solve2(u[0], u[1], psi)
    y = -1 * np.ones(shape=(num_pairs, 2))
    y[:, 0] = (np.random.uniform(size=num_pairs) < u[0]).astype(int)
    y[:, 1] = y[:, 0] * (np.random.uniform(size=num_pairs) <= u12 / u[0]) + (
        1 - y[:, 0]
    ) * (np.random.uniform(size=num_pairs) <= (u[1] - u12) / (1 - u[0]))
    return y


def correlated_binary_outcomes_from_uniforms(unifs, u, psi):
    """Generates correlated binary outcomes from uniform random numbers.

    Args:
        unifs (numpy.ndarray): An array of shape (n, 3) of uniform random
            numbers.
        u (list or tuple): A 2-item list or tuple of event probabilities.
        psi (float): The odds ratio of the binary outcomes.

    Returns:
        numpy.ndarray: A 2D array of paired binary outcomes.
    """
    if unifs.ndim == 2 and unifs.shape[1] == 3:
        u12 = _correlated_binary_outcomes_solve2(u[0], u[1], psi)
        n = unifs.shape[0]
        y = -1 * np.ones(shape=(n, 2))
        y[:, 0] = (unifs[:, 0] < u[0]).astype(int)
        y[:, 1] = y[:, 0] * (unifs[:, 1] <= u12 / u[0]) + (1 - y[:, 0]) * (
            unifs[:, 2] <= (u[1] - u12) / (1 - u[0])
        )
        return y
    else:
        raise ValueError("unifs must be an n*3 array")


def _create_conf_int_report(conf_int, alpha, method_name):
    """Creates a confidence interval report dictionary."""
    report = OrderedDict()
    report["Lower"] = conf_int[0]
    report["Upper"] = conf_int[1]
    report["Alpha"] = alpha
    report["Method"] = method_name
    return report


def get_proportion_confint_report(
    num_successes,
    num_trials,
    alpha=0.05,
    do_normal=True,
    do_agresti_coull=True,
    do_beta=False,
    do_wilson=True,
    do_jeffrey=False,
    do_binom_test=False,
):
    """Gets a report of confidence intervals for a proportion.

    Args:
        num_successes (int): The number of successes.
        num_trials (int): The number of trials.
        alpha (float, optional): The significance level. Defaults to 0.05.
        do_normal (bool, optional): If `True`, includes the normal
            approximation interval. Defaults to `True`.
        do_agresti_coull (bool, optional): If `True`, includes the
            Agresti-Coull interval. Defaults to `True`.
        do_beta (bool, optional): If `True`, includes the Clopper-Pearson
            (beta) interval. Defaults to `False`.
        do_wilson (bool, optional): If `True`, includes the Wilson score
            interval. Defaults to `True`.
        do_jeffrey (bool, optional): If `True`, includes Jeffrey's Bayesian
            interval. Defaults to `False`.
        do_binom_test (bool, optional): If `True`, includes the binomial test
            interval. Defaults to `False`.

    Returns:
        collections.OrderedDict: A dictionary of confidence interval reports.
    """
    from statsmodels.stats.proportion import proportion_confint

    conf_int_reports = OrderedDict()
    methods = {
        "Normal": ("normal", do_normal),
        "AgrestiCoull": ("agresti_coull", do_agresti_coull),
        "Beta": ("beta", do_beta),
        "Wilson": ("wilson", do_wilson),
        "Jeffrey": ("jeffrey", do_jeffrey),
        "BinomTest": ("binom_test", do_binom_test),
    }

    for report_name, (method_name, do_method) in methods.items():
        if do_method:
            conf_int = proportion_confint(
                num_successes, num_trials, alpha=alpha, method=method_name
            )
            conf_int_reports[report_name] = _create_conf_int_report(
                conf_int, alpha, report_name
            )

    return conf_int_reports


def cross_tab(
    col_row_pairs, cols=None, rows=None, to_json=False, do_value_counts=False
):
    """Creates a cross-tabulation of data pairs.

    Args:
        col_row_pairs (list[tuple]): A list of (column, row) pairs.
        cols (list, optional): A list of column headers. If `None`, the
            distinct items from the data are used. Defaults to `None`.
        rows (list, optional): A list of row headers. If `None`, the distinct
            items from the data are used. Defaults to `None`.
        to_json (bool, optional): If `True`, returns a JSON-friendly object.
            Defaults to `False`.
        do_value_counts (bool, optional): If `True`, returns value counts.
            Defaults to `False`.

    Returns:
        dict or pandas.DataFrame: A cross-tabulation of the data.
    """
    col_data, row_data = zip(*col_row_pairs)
    row_h = rows if rows else list(set(row_data))
    col_h = cols if cols else list(set(col_data))
    counts = np.zeros((len(row_h), len(col_h)))
    for i, r in enumerate(row_h):
        for j, c in enumerate(col_h):
            n = sum(
                np.array([x == c for x in col_data])
                & np.array([x == r for x in row_data])
            )
            counts[i, j] = n
    import pandas as pd

    df_n = pd.DataFrame(counts, index=row_h, columns=col_h)

    if not rows:
        row_order = np.argsort(-df_n.sum(axis=1).values)
        df_n = df_n.iloc[row_order]

    if to_json:
        return df_to_json(df_n, do_value_counts=do_value_counts)
    else:
        return df_n


class Memoize:
    """A class to cache function results."""

    def __init__(self, f):
        """Initializes a Memoize object.

        Args:
            f (callable): The function to memoize.
        """
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        """Calls the memoized function.

        Args:
            *args: The arguments to the function.

        Returns:
            The result of the function call.
        """
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


class ParameterSpace:
    """A class to handle combinations of parameters in simulations."""

    def __init__(self):
        """Initializes a ParameterSpace object."""
        self.vals_map = OrderedDict()

    def add(self, label, values):
        """Adds a parameter and its possible values to the space.

        Args:
            label (str): The name of the parameter.
            values (list): A list of possible values for the parameter.
        """
        self.vals_map[label] = values

    def sample(self, label):
        """Randomly samples a value for a given parameter.

        Args:
            label (str): The name of the parameter.

        Returns:
            object: A randomly sampled value.
        """
        if label in self.vals_map:
            vals = self.vals_map[label]
            return vals[np.random.choice(range(len(vals)))]
        else:
            return None

    def sample_all(self):
        """Randomly samples a value for each parameter.

        Returns:
            dict: A dictionary mapping parameter names to sampled values.
        """
        sampled = {}
        for label in self.vals_map:
            sampled[label] = self.sample(label)
        return sampled

    def get_cyclical_iterator(self, limit=-1):
        """Gets a cyclical iterator for the parameter space.

        Args:
            limit (int, optional): The maximum number of iterations.
                -1 for infinite. Defaults to -1.

        Returns:
            _ParameterSpaceIter: An iterator for the parameter space.
        """
        return _ParameterSpaceIter(self, limit)

    def keys(self):
        """Gets the names of the parameters.

        Returns:
            list: A list of parameter names.
        """
        return self.vals_map.keys()

    def dimensions(self):
        """Gets the number of values for each parameter.

        Returns:
            numpy.ndarray: An array of the number of values for each
                parameter.
        """
        return np.array([len(y) for x, y in self.vals_map.items()])

    def size(self):
        """Gets the total size of the parameter space.

        Returns:
            int: The total number of parameter combinations.
        """
        return np.prod(self.dimensions())

    def __getitem__(self, key):
        """Gets the values for a given parameter.

        Args:
            key (str): The name of the parameter.

        Returns:
            list: The list of values for the parameter.
        """
        return self.vals_map[key]


class _ParameterSpaceIter:
    """An iterator for the ParameterSpace class."""

    def __init__(self, parameter_space, limit):
        """Initializes a _ParameterSpaceIter object."""
        self.limit = limit
        self.cursor = 0
        self.vals_map = copy(parameter_space.vals_map)
        self.labels = list(self.vals_map.keys())
        num_options = []
        for label in self.labels:
            num_options.append(len(parameter_space[label]))
        self.paths = list(product(*[range(x) for x in num_options]))

    def __iter__(self):
        return self

    def __next__(self):
        if 0 < self.limit <= self.cursor:
            raise StopIteration()
        i = self.cursor % len(self.paths)
        path = self.paths[i]
        param_map = {}
        assert len(path) == len(self.labels)
        for j, label in enumerate(self.labels):
            param_map[label] = self.vals_map[label][path[j]]
        self.cursor += 1
        return param_map

    next = __next__


if __name__ == "__main__":
    import doctest

    doctest.testmod()
