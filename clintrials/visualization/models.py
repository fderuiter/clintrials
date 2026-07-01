from dataclasses import dataclass
import pandas as pd
import numpy as np

def _format_label(label):
    if not isinstance(label, str):
        return label
    return label.replace("_", " ").title()

@dataclass
class TextSection:
    text: str

    def __str__(self):
        return self.text

@dataclass
class TableSection:
    title: str
    df: pd.DataFrame

    def __str__(self):
        """Generates a text summary for a chart based on its dataframe."""
        summary = f"**Data Summary: {self.title}**\n\n"
        cols = list(self.df.columns)

        def fmt(v):
            if isinstance(v, (float, np.float64)):
                return f"{v:.4f}"
            return str(v)

        header = "| " + " | ".join([_format_label(c) for c in cols]) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"

        rows = []
        for _, row in self.df.iterrows():
            rows.append("| " + " | ".join([fmt(x) for x in row]) + " |")

        return summary + "\n".join([header, sep] + rows)
