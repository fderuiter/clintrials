import json
import warnings
from collections.abc import Iterable
from datetime import datetime
import numpy as np

def to_1d_list_gen(x):
    """Generator function to flatten a list of lists.
    
    .. deprecated:: 0.1.4
       Use `to_1d_list` or an alternative.
    """
    warnings.warn(
        "to_1d_list_gen is deprecated",
        DeprecationWarning,
    )
    if isinstance(x, list):
        for y in x:
            yield from to_1d_list_gen(y)
    else:
        yield x

def to_1d_list(x):
    """Flattens a list of lists of arbitrary depth to a single list.
    
    .. deprecated:: 0.1.4
       Use standard list comprehensions instead.
    """
    warnings.warn(
        "to_1d_list is deprecated",
        DeprecationWarning,
    )
    return list(to_1d_list_gen(x))

def _open_json_url(url):
    """Opens a JSON file from a URL.
    
    .. deprecated:: 0.1.4
       Use modern libraries like `requests` instead.
    """
    warnings.warn(
        "_open_json_url is deprecated",
        DeprecationWarning,
    )
    from urllib.request import urlopen
    return json.load(urlopen(url))

def fullname(o):
    """Gets the fully-qualified class name of an object.
    
    .. deprecated:: 0.1.4
       No modern alternative provided.
    """
    warnings.warn(
        "fullname is deprecated",
        DeprecationWarning,
    )
    return o.__module__ + "." + o.__class__.__name__

def atomic_to_json(obj):
    """Converts an atomic object to a JSON-friendly format.
    
    .. deprecated:: 0.1.4
       No modern alternative provided.
    """
    warnings.warn(
        "atomic_to_json is deprecated",
        DeprecationWarning,
    )
    if isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def iterable_to_json(obj):
    """Converts an iterable to a JSON-friendly list.
    
    .. deprecated:: 0.1.4
       No modern alternative provided.
    """
    warnings.warn(
        "iterable_to_json is deprecated",
        DeprecationWarning,
    )
    if isinstance(obj, Iterable):
        return [atomic_to_json(x) for x in obj]
    else:
        return atomic_to_json(obj)

def row_to_json(row, **kwargs):
    """Converts a pandas Series to a JSON-friendly dictionary.
    
    .. deprecated:: 0.1.4
       No modern alternative provided.
    """
    warnings.warn(
        "row_to_json is deprecated",
        DeprecationWarning,
    )
    try:
        doc = json.loads(row.to_json(), **kwargs)
    except UnicodeDecodeError:
        return row_to_json(row, encoding="iso-8859-1")
    import pandas as pd
    for x in row.index:
        if isinstance(row[x], datetime) and not pd.isnull(row[x]):
            doc[x] = pd.to_datetime(row[x]).date().isoformat()
    return doc

