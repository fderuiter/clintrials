from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clintrials.visualization.helpers import format_label as _format_label
from clintrials.visualization.helpers import format_number as fmt


@dataclass
class TextSection:
    text: str

    def __str__(self):  # type: ignore
        return self.text

@dataclass
class MultiFormatSummaryContainer:
    title: str
    df: pd.DataFrame

    @property
    def markdown(self):  # type: ignore
        """Generates a text summary for a chart based on its dataframe."""
        summary = f"**Data Summary: {self.title}**\n\n"
        cols = list(self.df.columns)

        header = "| " + " | ".join([_format_label(c) for c in cols]) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"

        rows = []
        for _, row in self.df.iterrows():
            rows.append("| " + " | ".join([fmt(x) for x in row]) + " |")

        return summary + "\n".join([header, sep] + rows)

    def __str__(self):  # type: ignore
        return self.markdown

    @property
    def html(self):  # type: ignore
        """Generates an accessible HTML summary table."""
        try:
            import streamlit as st
            acc_mode = st.session_state.get('accessibility_mode', False)
        except ImportError:
            acc_mode = False

        if acc_mode:
            return self._generate_hierarchical_html()  # type: ignore

        summary = f"<strong>Data Summary: {self.title}</strong><br><br>\n"
        cols = list(self.df.columns)

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

    def _generate_hierarchical_html(self):  # type: ignore
        """Generates a hierarchical accessible HTML summary table using nested details."""
        summary = f"<strong>Data Summary: {self.title}</strong><br><br>\n"

        # Add Expand All / Collapse All buttons
        summary += """<div style="margin-bottom: 10px;">
  <button type="button" onclick="document.querySelectorAll('details').forEach(e => e.setAttribute('open', 'true'))" aria-label="Expand all data sections">Expand All</button>
  <button type="button" onclick="document.querySelectorAll('details').forEach(e => e.removeAttribute('open'))" aria-label="Collapse all data sections">Collapse All</button>
</div>\n"""

        # Determine grouping columns
        target_cols = [c for c in self.df.columns if c.lower() in ['trial', 'cohort', 'dose', 'group', 'arm', 'scenario']]
        if len(target_cols) > 0:
            grouping_cols = [c for c in self.df.columns if c in target_cols]
        else:
            # Fallback to first few categorical/low-cardinality columns
            grouping_cols = []
            cat_cols = set(self.df.select_dtypes(include=['object', 'string', 'category']).columns)
            for c in self.df.columns:
                if c in cat_cols or self.df[c].nunique() < len(self.df) / 2:
                    if len(grouping_cols) < 3:
                        grouping_cols.append(c)
            # Ensure we don't group by all columns
            if len(grouping_cols) == len(self.df.columns):
                grouping_cols = grouping_cols[:-1]

        # Limit to 3 levels of nesting (4 levels total including leaf table)
        grouping_cols = grouping_cols[:3]

        def _build_html_table(df):
            cols = list(df.columns)
            html = "<table>\n  <thead>\n    <tr>\n"
            for c in cols:
                html += f'      <th scope="col">{_format_label(c)}</th>\n'
            html += "    </tr>\n  </thead>\n  <tbody>\n"
            for _, row in df.iterrows():
                html += "    <tr>\n"
                for x in row:
                    html += f"      <td>{fmt(x)}</td>\n"  # type: ignore
                html += "    </tr>\n"
            html += "  </tbody>\n</table>\n"
            return html

        if not grouping_cols:
            return summary + _build_html_table(self.df)  # type: ignore

        def generate_level(df, current_grouping_cols, level=1):  # type: ignore
            if not current_grouping_cols:
                return _build_html_table(df)  # type: ignore

            col = current_grouping_cols[0]
            grouped = df.groupby(col)

            html = ""
            for name, group in grouped:
                # Calculate summaries for numeric columns
                numeric_cols = group.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    metrics = []
                    metrics.append(f"N={len(group)}")
                    for nc in numeric_cols:
                        if nc not in grouping_cols:
                            metrics.append(f"Mean {_format_label(nc)}: {fmt(group[nc].mean())}")  # type: ignore
                    summary_str = " | ".join(metrics[:4]) # limit to 4 metrics to avoid verbosity
                else:
                    summary_str = f"N={len(group)}"

                # ARIA disclosure pattern is handled natively by <details> and <summary> tags
                heading_level = min(level + 2, 6) # e.g. h3, h4, h5
                html += '<details>\n'
                html += f'  <summary style="cursor: pointer;"><h{heading_level} style="display: inline; margin: 0; font-size: 1em;">{_format_label(col)}: {name}</h{heading_level}> <span style="font-size: 0.9em; color: #555;">({summary_str})</span></summary>\n'
                html += f'  <div style="margin-left: {20 * level}px; margin-top: 10px; margin-bottom: 10px;">\n'
                html += generate_level(group.drop(columns=[col]), current_grouping_cols[1:], level + 1)  # type: ignore
                html += '  </div>\n'
                html += '</details>\n'
            return html

        html = summary + generate_level(self.df, grouping_cols)  # type: ignore
        return html

    def to_plotly_json(self):  # type: ignore
        """Serialize this object to a JSON dictionary for Plotly."""
        return {"title": self.title, "markdown": self.markdown, "html": self.html}
