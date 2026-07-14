import numpy as np

def format_label(label):
    """Format a label by replacing underscores with spaces and applying title case."""
    if not isinstance(label, str):
        return label
    return label.replace("_", " ").title()

def format_labels_dict(cols):
    """Create a dictionary mapping column names to their formatted labels."""
    if not isinstance(cols, list):
        cols = [cols]
    labels = {}
    for col in cols:
        if isinstance(col, str):
            labels[col] = format_label(col)
    return labels

def format_number(v):
    """Format numeric values to 4 decimal places, or cast to string if not float."""
    if isinstance(v, (float, np.float64)):
        return f"{v:.4f}"
    return str(v)
