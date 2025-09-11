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
    """Reduces lists of lists of arbitrary depth to a single depth-1 list.

    Note:
        This function is recursive.

    Args:
        x: A scalar, list, or list of lists.

    Yields:
        An element of the flattened list.
    """

    if isinstance(x, list):
        for y in x:
            yield from to_1d_list_gen(y)
    else:
        yield x


def to_1d_list(x):
    """Reshapes scalars, lists, and lists of lists to a single flat list.

    Note:
        This function uses a generator function to flatten the list.

    Args:
        x: A scalar, list, or list of lists.

    Returns:
        A flattened list.

    Examples:
        >>> to_1d_list(0)
        [0]
        >>> to_1d_list([1])
        [1]
        >>> to_1d_list([[1,2],3,[4,5]])
        [1, 2, 3, 4, 5]
        >>> to_1d_list([[1,2],3,[4,5,[6,[7,8,[9]]]]])
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return list(to_1d_list_gen(x))


def get_logger(name: str = __name__) -> logging.Logger:
    """Gets a logger instance.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)


def _open_json_local(file_loc):
    return json.load(open(file_loc))


def _open_json_url(url):
    from urllib.request import urlopen

    return json.load(urlopen(url))


def fetch_json_from_files(file_pattern):
    """Fetches and combines JSON data from multiple files.

    Args:
        file_pattern: A file pattern (e.g., "data/*.json") to match files.

    Returns:
        A list containing the combined JSON data from all matched files.
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
        list_of_dicts: A list of dictionaries to filter.
        filter_dict: A dictionary of key-value pairs to filter by.
            Only exact matches are retained.

    Returns:
        A new list of dictionaries containing only the filtered items.
    """
    for key, val in filter_dict.items():
        # In JSON, tuples are masked as lists. In this filter, we treat them as equivalent:
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
        files: A list of files to process.
        map_func: A function to apply to each file.
        reduce_func: A function to reduce the results of the map function.

    Returns:
        The result of the map-reduce operation.

    Raises:
        TypeError: If the list of files is empty.
    """
    if len(files):
        x = map(map_func, files)
        return reduce(reduce_func, x)
    else:
        raise TypeError("No files")


def invoke_map_reduce_on_list(a_list, function_map):
    """Invokes a map-reduce pattern for many items on a list.

    Functions are specified as "item name" -> (map_func, reduce_func) pairs
    in function_map. In each iteration, map_func is invoked on sims, and then
    reduce_func is invoked on the result.

    Args:
        a_list: The list to process.
        function_map: A dictionary where keys are item names and values are
            tuples of (map_func, reduce_func).

    Returns:
        An ordered dictionary with keys from function_map and values as the
        result of the reduce function.
    """

    response = OrderedDict()
    for item, function_tuple in function_map.items():
        map_func, reduce_func = function_tuple
        x = reduce(reduce_func, map(map_func, a_list))
        response[item] = x

    return response


def reduce_maps_by_summing(x, y):
    """Reduces two maps by summing the values of matching keys.

    Args:
        x: The first map (dictionary).
        y: The second map (dictionary).

    Returns:
        A new dictionary with the summed values.
    """

    response = OrderedDict()
    for k in x.keys():
        response[k] = x[k] + y[k]
    return response


def multiindex_dataframe_from_tuple_map(x, labels):
    """Creates a pandas DataFrame with a MultiIndex from a map.

    Args:
        x: A map of parameter-tuple -> value pairs.
        labels: A list of labels for the MultiIndex.

    Returns:
        A pandas DataFrame with a MultiIndex.
    """
    import pandas as pd

    k, v = zip(*[(k, v) for (k, v) in x.items()])
    i = pd.MultiIndex.from_tuples(k, names=labels)
    return pd.DataFrame(list(v), index=i)


def fullname(o):
    """Gets the fully-qualified class name of an object.

    Args:
        o: An object of any kind.

    Returns:
        The fully-qualified class name of the object.
    """

    return o.__module__ + "." + o.__class__.__name__


def atomic_to_json(obj):
    """Converts an object to a JSON-friendly format.

    Note:
        This function is particularly useful for converting numpy types that
        are not directly serializable to JSON.

    Args:
        obj: The object to convert.

    Returns:
        The object in a JSON-serializable format.
    """

    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def iterable_to_json(obj):
    """Converts an iterable of objects to a JSON-friendly list.

    Args:
        obj: An iterable of objects to convert.

    Returns:
        A list of JSON-serializable objects.
    """
    if isinstance(obj, Iterable):
        return [atomic_to_json(x) for x in obj]
    else:
        return atomic_to_json(obj)


def row_to_json(row, **kwargs):
    """Converts a pandas Series to a JSON object.

    Args:
        row: The pandas Series to convert.
        **kwargs: Keyword arguments to pass to `json.loads`.

    Returns:
        A JSON-friendly dictionary representation of the row.
    """

    try:
        doc = json.loads(row.to_json(), **kwargs)
    except UnicodeDecodeError:
        # iso-8859-1 has solved this before; but might not solve all ills.
        return row_to_json(row, encoding="iso-8859-1")
    # to_json turns all dates to long as 'ticks after epoch',
    # regardless of params passed (bug?) so cast dates to isoformat manually:
    # n.b. only actual dates can be cast to an isoformat string so screen null dates.
    import pandas as pd

    for x in row.index:
        if isinstance(row[x], datetime) and not pd.isnull(row[x]):
            doc[x] = pd.to_datetime(row[x]).date().isoformat()
    return doc


def _serialize_table_structure(df):
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
    freqs = OrderedDict()
    for col_name in df:
        vc = df[col_name].value_counts()
        if len(vc) < len(df) or definitely_do_value_counts:
            freqs[atomic_to_json(col_name)] = {
                atomic_to_json(k): atomic_to_json(v) for k, v in vc.items()
            }
    return freqs


def _calculate_column_summaries(df):
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
    """Serializes a pandas DataFrame to a JSON-able object.

    Note:
        pandas.DataFrame provides its own JSON serialization method, but this
        function provides a custom format.

    Args:
        df: The DataFrame to serialize.
        do_value_counts: Whether to calculate value counts for each column.
        definitely_do_value_counts: If True, forces value count aggregation
            even if all elements are unique.
        do_column_summaries: Whether to calculate summary statistics for
            each column.
        do_row_summaries: Whether to calculate summary statistics for each row.

    Returns:
        A JSON-able dictionary representation of the DataFrame.
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

    See http://en.wikipedia.org/wiki/Levenshtein_distance for more details.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Levenshtein distance between the two strings.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row
            # are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_index(s1, s2):
    """Calculates a similarity score between two strings (0 to 1).

    This method uses the Levenshtein distance and normalizes it by the
    length of the longer string.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        A similarity index between 0.0 and 1.0.
    """

    l = levenshtein(s1, s2)
    max_length = max(len(s1), len(s2))
    if max_length:
        return 1 - 1.0 * l / max_length
    else:
        return 0.0


def support_match(a, b):
    """Calculates a percentage score of the overlap between two collections.

    The score represents the proportion of elements in `a` that are also in
    `b`, plus the proportion of elements in `b` that are also in `a`.

    Args:
        a: The first collection.
        b: The second collection.

    Returns:
        A match score between 0.0 and 1.0.
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
    """Helper function to correlated_binary_outcomes"""
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
    if psi == 1:
        return mui * muj
    else:
        a = 1 - psi
        b = 1 - a * (mui + muj)
        c = -psi * (mui * muj)
        muij = _correlated_binary_outcomes_mardia(a, b, c)
    return muij


def correlated_binary_outcomes(num_pairs, u, psi, seed=None):
    """Randomly samples correlated binary outcomes.

    This method is based on the R package `ranBin2`.

    Note:
        For details on estimating correlation from odds ratio and vice-versa,
        see Yule's method (1912), as described in the Bonett article at
        http://psych.colorado.edu/~willcutt/pdfs/Bonett_2007.pdf

    Args:
        num_pairs: The number of pairs to sample.
        u: A 2-item list or tuple of event probabilities.
        psi: The odds ratio of the binary outcomes.
        seed: An optional seed for reproducible randomness.

    Returns:
        A numpy.ndarray of paired binary digits, with 2 columns and
        `num_pairs` rows.
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
    """Creates correlated binary outcomes from a given array of uniforms.

    This method is a tweaked version of the method from the R package
    `ranBin2`.

    Note:
        For details on estimating correlation from odds ratio and vice-versa,
        see Yule's method (1912), as described in the Bonett article at
        http://psych.colorado.edu/~willcutt/pdfs/Bonett_2007.pdf

    Args:
        unifs: A numpy.ndarray of shape (n, 3) of uniforms between 0 and 1.
        u: A 2-item list or tuple of event probabilities.
        psi: The odds ratio of the binary outcomes.

    Returns:
        A numpy.ndarray of paired binary digits, with 2 columns and n rows.

    Raises:
        ValueError: If `unifs` is not an n*3 array.
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
    """Gets confidence intervals for a binomial proportion.

    This function provides a report of confidence intervals calculated using
    various methods from `statsmodels`.

    Args:
        num_successes: The number of successes.
        num_trials: The total number of trials.
        alpha: The significance level (e.g., 0.05 for a 95% CI).
        do_normal: Whether to include the normal approximation interval.
        do_agresti_coull: Whether to include the Agresti-Coull interval.
        do_beta: Whether to include the Clopper-Pearson interval (beta).
        do_wilson: Whether to include the Wilson score interval.
        do_jeffrey: Whether to include Jeffrey's Bayesian interval.
        do_binom_test: Whether to include the binomial test interval.

    Returns:
        An ordered dictionary containing the reports for each calculated
        confidence interval.
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
    """Cross-tabulates counts of data pairs.

    Args:
        col_row_pairs: A list of 2-tuples, where each tuple is
            (column_item, row_item).
        cols: A list of column headers. If omitted, distinct items from
            `col_row_pairs` will be used.
        rows: A list of row headers. If omitted, distinct items from
            `col_row_pairs` will be used, and rows will be sorted by
            row-wise totals.
        to_json: If True, returns a JSON-able object; otherwise, returns a
            pandas DataFrame.
        do_value_counts: If True, includes value counts in the JSON output.

    Returns:
        A cross-tabulation in the form of a pandas DataFrame or a
        JSON-able dictionary.
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
    """A class to cache function results based on their arguments.

    Examples:
        >>> f = lambda x: x**3
        >>> f = Memoize(f)
        >>> f(2.0)  # Result is calculated and cached
        8.0
        >>> f(2.0)  # Result is fetched from cache
        8.0
    """

    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


class ParameterSpace:
    """A class to handle combinations of parameters in simulations."""

    def __init__(self):
        self.vals_map = OrderedDict()

    def add(self, label, values):
        """Adds a parameter and its possible values.

        Args:
            label: The name of the parameter.
            values: A list of possible values for the parameter.
        """

        self.vals_map[label] = values

    def sample(self, label):
        """Randomly samples a value for a given parameter.

        Args:
            label: The name of the parameter to sample.

        Returns:
            A randomly sampled value for the parameter, or None if the
            parameter does not exist.
        """

        if label in self.vals_map:
            vals = self.vals_map[label]
            return vals[np.random.choice(range(len(vals)))]
        else:
            return None

    def sample_all(self):
        """Randomly samples a value for each parameter.

        Returns:
            A dictionary mapping each parameter to a randomly sampled value.
        """

        sampled = {}
        for label in self.vals_map:
            sampled[label] = self.sample(label)
        return sampled

    def get_cyclical_iterator(self, limit=-1):
        """Gets an iterator to cycle through all parameter permutations.

        Args:
            limit: The maximum number of permutations to iterate through. If -1,
                the iterator will cycle forever.

        Returns:
            An iterable object that yields parameter permutations.
        """

        return _ParameterSpaceIter(self, limit)

    def keys(self):
        """Gets the names of all parameters in the space.

        Returns:
            An iterable of parameter names.
        """

        return self.vals_map.keys()

    def dimensions(self):
        """Gets the number of values for each parameter.

        Returns:
            A numpy array containing the number of values for each parameter.
        """

        return np.array([len(y) for x, y in self.vals_map.items()])

    def size(self):
        """Gets the total size of the parameter space.

        The size is the product of the number of values for each parameter.

        Returns:
            The size of the parameter space.
        """

        return np.prod(self.dimensions())

    def __getitem__(self, key):
        return self.vals_map[key]


class _ParameterSpaceIter:

    def __init__(self, parameter_space, limit):
        self.limit = limit
        self.cursor = 0
        self.vals_map = copy(parameter_space.vals_map)
        self.labels = list(self.vals_map.keys())
        num_options = []
        for label in self.labels:
            num_options.append(len(parameter_space[label]))
        self.paths = list(product(*[range(x) for x in num_options]))
        # print zip(labels, num_options)

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

    # Python 2 compatibility alias (deprecated)
    next = __next__


if __name__ == "__main__":
    import doctest

    doctest.testmod()
