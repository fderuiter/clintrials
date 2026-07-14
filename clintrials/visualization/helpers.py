import numpy as np

def format_label(label):
    if not isinstance(label, str):
        return label
    return label.replace("_", " ").title()

def format_labels_dict(cols):
    if not isinstance(cols, list):
        cols = [cols]
    labels = {}
    for col in cols:
        if isinstance(col, str):
            labels[col] = format_label(col)
    return labels

def format_number(v):
    if isinstance(v, (float, np.float64)):
        return f"{v:.4f}"
    return str(v)
