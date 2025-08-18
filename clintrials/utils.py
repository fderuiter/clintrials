__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

import glob
import json
import logging
from collections import OrderedDict
from functools import reduce

logger = logging.getLogger(__name__)


def _open_json_local(file_loc):
    return json.load(open(file_loc))


def _open_json_url(url):
    from urllib.request import urlopen

    return json.load(urlopen(url))


def fetch_json_from_files(file_pattern):
    files = glob.glob(file_pattern)
    sims = []
    for f in files:
        sub_sims = _open_json_local(f)
        logger.info("%s %s", f, len(sub_sims))
        sims += sub_sims
    logger.info("Fetched %s sims", len(sims))
    return sims


def filter_list_of_dicts(list_of_dicts, filter_dict):
    """Filter a list of dictionaries.

    :param list_of_dicts: list of dictionaries
    :type list_of_dicts: list
    :param filter_dict: map of item -> value pairs that forms the filter. Exact matches are retained.
    :type filter_dict: dict

    """
    for key, val in filter_dict.items():
        # In JSON, tuples are masked as lists. In this filter, we treat them as equivalent:
        if isinstance(val, (tuple)):
            list_of_dicts = [x for x in list_of_dicts if x[key] == val or x[key] == list(val)]
        else:
            list_of_dicts = [x for x in list_of_dicts if x[key] == val]
    return list_of_dicts


def map_reduce_files(files, map_func, reduce_func):
    """
    Invoke map_func on each file in sim_files and reduce results using reduce_func.

    :param files: list of files that contain simulations
    :type files: list
    :param map_func:function to create summary content for object x
    :type map_func: function
    :param reduce_func: function to reduce summary content of objects x & y
    :type reduce_func: function

    :returns: ?
    :rtype: ?

    """
    if len(files):
        x = map(map_func, files)
        return reduce(reduce_func, x)
    else:
        raise TypeError("No files")


def invoke_map_reduce_on_list(a_list, function_map):
    """Invokes map/reduce pattern for many items on a list.
    Functions are specified as "item name" -> (map_func, reduce_func) pairs in function_map.
    In each iteration, map_func is invoked on sims, and then reduce_func is invoked on result.
    As usual, map_func takes iterable as single argument and reduce_func takes x and y as args.

    Returns a dict with keys function_map.keys() and values the result of reduce_func
    """

    response = OrderedDict()
    for item, function_tuple in function_map.items():
        map_func, reduce_func = function_tuple
        x = reduce(reduce_func, map(map_func, a_list))
        response[item] = x

    return response


def reduce_maps_by_summing(x, y):
    """Reduces maps x and y by adding the value of every item in x to matching value in y.

    :param x: first map
    :type x: dict
    :param y: second map
    :type y: dict
    :returns: map of summed values
    :rtype: dict

    """

    response = OrderedDict()
    for k in x.keys():
        response[k] = x[k] + y[k]
    return response


def multiindex_dataframe_from_tuple_map(x, labels):
    """Create pandas.DataFrame from map of param-tuple -> value

    :param x: map of parameter-tuple -> value pairs
    :type x: dict
    :param labels: list of item labels
    :type labels: list
    :returns: DataFrame object
    :rtype: pandas.DataFrame

    """
    import pandas as pd

    k, v = zip(*[(k, v) for (k, v) in x.items()])
    i = pd.MultiIndex.from_tuples(k, names=labels)
    return pd.DataFrame(list(v), index=i)
