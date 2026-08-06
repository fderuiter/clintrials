# SPDX-License-Identifier: MIT

"""Helper visualization utilities."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def format_label(label: str | Any) -> str | Any:
    """Format a label by replacing underscores with spaces and applying title case.

    Args:
        label (str or Any): The raw label string or value.

    Returns:
        str or Any: The formatted label string, or the input value if not a string.
    """
    if not isinstance(label, str):
        return label
    return label.replace("_", " ").title()

def format_labels_dict(cols: str | List[str]) -> Dict[str, str]:
    """Create a dictionary mapping column names to their formatted labels.

    Args:
        cols (str or List[str]): A single column name or a list of column names.

    Returns:
        Dict[str, str]: A dictionary mapping the input columns to their formatted labels.
    """
    if not isinstance(cols, list):
        cols = [cols]
    labels = {}
    for col in cols:
        if isinstance(col, str):
            labels[col] = format_label(col)
    return labels

def format_number(v: float | int | Any) -> str:
    """Format numeric values to 4 decimal places, or cast to string if not float.

    Args:
        v (float or int or Any): The numerical or arbitrary value to format.

    Returns:
        str: The formatted string.
    """
    if isinstance(v, (float, np.float64)):
        return f"{v:.4f}"
    return str(v)

def build_html_table(df: pd.DataFrame) -> str:
    """Build a standard HTML table from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to render into HTML.

    Returns:
        str: The generated HTML table markup.
    """
    cols = list(df.columns)
    html = "<table>\n"
    html += "  <thead>\n    <tr>\n"
    for c in cols:
        html += f'      <th scope="col">{format_label(c)}</th>\n'
    html += "    </tr>\n  </thead>\n"

    html += "  <tbody>\n"
    for _, row in df.iterrows():
        html += "    <tr>\n"
        for x in row:
            html += f"      <td>{format_number(x)}</td>\n"
        html += "    </tr>\n"
    html += "  </tbody>\n</table>"
    return html
