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
class MultiFormatSummaryContainer:
    title: str
    df: pd.DataFrame

    @property
    def markdown(self):
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

    def __str__(self):
        return self.markdown

    @property
    def html(self):
        """Generates an accessible HTML summary table."""
        summary = f"<strong>Data Summary: {self.title}</strong><br><br>\n"
        cols = list(self.df.columns)

        def fmt(v):
            if isinstance(v, (float, np.float64)):
                return f"{v:.4f}"
            return str(v)

        html = summary + "<table>\n"
        html += "  <thead>\n    <tr>\n"
        for c in cols:
            html += f'      <th scope="col">{_format_label(c)}</th>\n'
        html += "    </tr>\n  </thead>\n"
        
        html += "  <tbody>\n"
        for _, row in self.df.iterrows():
            html += "    <tr>\n"
            for x in row:
                html += f"      <td>{fmt(x)}</td>\n"
            html += "    </tr>\n"
        html += "  </tbody>\n</table>"
        
        return html

    def to_plotly_json(self):
        """Serialize this object to a JSON dictionary for Plotly."""
        return {"title": self.title, "markdown": self.markdown, "html": self.html}
