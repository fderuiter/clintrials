from dataclasses import dataclass
import html
import pandas as pd
import numpy as np

from clintrials.visualization.helpers import format_label as _format_label, format_number as fmt

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
        try:
            import streamlit as st
            acc_mode = st.session_state.get('accessibility_mode', False)
        except ImportError:
            acc_mode = False

        if acc_mode:
            return self._generate_hierarchical_html()

        safe_title = html.escape(str(self.title), quote=True)
        summary = f"<strong>Data Summary: {safe_title}</strong><br><br>\n"
        cols = list(self.df.columns)

        res = summary + "<table>\n"
        res += "  <thead>\n    <tr>\n"
        for c in cols:
            safe_label = html.escape(str(_format_label(c)), quote=True)
            res += f'      <th scope="col">{safe_label}</th>\n'
        res += "    </tr>\n  </thead>\n"
        
        res += "  <tbody>\n"
        for _, row in self.df.iterrows():
            res += "    <tr>\n"
            for x in row:
                safe_val = html.escape(str(fmt(x)), quote=True)
                res += f"      <td>{safe_val}</td>\n"
            res += "    </tr>\n"
        res += "  </tbody>\n</table>"

        return res

    def _generate_hierarchical_html(self):
        """Generates a hierarchical accessible HTML summary table using nested details."""
        safe_title = html.escape(str(self.title), quote=True)
        summary = f"<strong>Data Summary: {safe_title}</strong><br><br>\n"

        # Add Expand All / Collapse All buttons
        summary += """<div style="margin-bottom: 10px;">
  <button type="button" onclick="document.querySelectorAll('details').forEach(e => e.setAttribute('open', 'true'))" aria-label="Expand all data sections">Expand All</button>
  <button type="button" onclick="document.querySelectorAll('details').forEach(e => e.removeAttribute('open'))" aria-label="Collapse all data sections">Collapse All</button>
</div>\n"""

        # Determine grouping columns
        target_cols = [c for c in self.df.columns if isinstance(c, str) and c.lower() in ['trial', 'cohort', 'dose', 'group', 'arm', 'scenario']]
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
            res = "<table>\n  <thead>\n    <tr>\n"
            for c in cols:
                safe_label = html.escape(str(_format_label(c)), quote=True)
                res += f'      <th scope="col">{safe_label}</th>\n'
            res += "    </tr>\n  </thead>\n  <tbody>\n"
            for _, row in df.iterrows():
                res += "    <tr>\n"
                for x in row:
                    safe_val = html.escape(str(fmt(x)), quote=True)
                    res += f"      <td>{safe_val}</td>\n"
                res += "    </tr>\n"
            res += "  </tbody>\n</table>\n"
            return res

        if not grouping_cols:
            return summary + _build_html_table(self.df)

        def generate_level(df, current_grouping_cols, level=1):
            if not current_grouping_cols:
                return _build_html_table(df)

            col = current_grouping_cols[0]
            grouped = df.groupby(col, dropna=False)

            res = ""
            for name, group in grouped:
                # Calculate summaries for numeric columns
                numeric_cols = group.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    metrics = []
                    metrics.append(f"N={len(group)}")
                    for nc in numeric_cols:
                        if nc not in grouping_cols:
                            safe_metric_label = html.escape(str(_format_label(nc)), quote=True)
                            metrics.append(f"Mean {safe_metric_label}: {fmt(group[nc].mean())}")
                    summary_str = " | ".join(metrics[:4]) # limit to 4 metrics to avoid verbosity
                else:
                    summary_str = f"N={len(group)}"

                # ARIA disclosure pattern is handled natively by <details> and <summary> tags
                heading_level = min(level + 2, 6) # e.g. h3, h4, h5
                res += '<details>\n'
                safe_col_label = html.escape(str(_format_label(col)), quote=True)
                safe_name = html.escape(str(name), quote=True)
                safe_summary_str = html.escape(str(summary_str), quote=True)
                res += f'  <summary style="cursor: pointer;"><h{heading_level} style="display: inline; margin: 0; font-size: 1em;">{safe_col_label}: {safe_name}</h{heading_level}> <span style="font-size: 0.9em; color: #555;">({safe_summary_str})</span></summary>\n'
                res += f'  <div style="margin-left: {20 * level}px; margin-top: 10px; margin-bottom: 10px;">\n'
                res += generate_level(group.drop(columns=[col]), current_grouping_cols[1:], level + 1)
                res += '  </div>\n'
                res += '</details>\n'
            return res

        res = summary + generate_level(self.df, grouping_cols)
        return res

    def to_plotly_json(self):
        """Serialize this object to a JSON dictionary for Plotly."""
        return {"title": self.title, "markdown": self.markdown, "html": self.html}
