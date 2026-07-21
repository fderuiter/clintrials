"""Renders the Win Ratio simulation view in the Streamlit dashboard.

Random Seed Strategy: {winratio_view_seed_strategy}
"""

import pandas as pd
import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.core.schema import WinRatioSchema
from clintrials.visualization.dashboard.factory import create_widget, render_metric
from clintrials.visualization.dashboard.views.framework import dashboard_view
from clintrials.winratio.main import run_winratio_simulations


@PROTOCOL_REGISTRY.register("Win Ratio")
@dashboard_view(
    title="Win Ratio Simulation",
    model_name="Win Ratio",
    file_prefix="winratio_simulation",
    csv_index=False,
    skip_summary_table=True
)
def render() -> None:
    """Render the Win Ratio simulation interface."""
    st.sidebar.header("Trial Parameters")

    # Use schema to generate UI inputs
    kwargs = {}
    for name, field in WinRatioSchema.model_fields.items():
        min_val = 0.0 if "Probability" in str(field.annotation) else 1
        max_val = 1.0 if "Probability" in str(field.annotation) else None

        kwargs[name] = create_widget(
            st,
            "number_input",
            name,
            field.description,
            min_value=min_val,
            max_value=max_val,
            value=field.default,
        )

    if create_widget(st, "button", "run_simulation_button", "Run Simulation"):
        from clintrials.visualization.dashboard.utils import announce_status_locally
        announce_status_locally("Simulation in progress", key="winratio-start")
        try:
            with st.spinner("Running simulation..."):
                summary = run_winratio_simulations(**kwargs)
                power = summary["power"]
                average_ci = summary["average_ci"]
            announce_status_locally("Simulation completed", key="winratio-complete")
            st.success("Simulation complete")
        except Exception as e:
            announce_status_locally("Simulation failed", key="winratio-fail")
            raise e

        st.subheader("Results")

        if not hasattr(st, "columns"):
            st.columns = lambda x: (st, st)
        met_col1, met_col2 = st.columns(2)

        render_metric(met_col1, "Power", power)
        render_metric(met_col2, "Average 95% Confidence Interval", average_ci)

        # Create simple DataFrame for export
        results_dict = kwargs.copy()
        results_dict["power"] = power
        results_dict["ci_lower"] = average_ci[0]
        results_dict["ci_upper"] = average_ci[1]

        df = pd.DataFrame([results_dict])

        import clintrials.visualization as viz
        fig = viz.plot_winratio_power_curve(
            df,
            high_contrast=False
        )
        figures = [(None, fig)]

        from clintrials.visualization.helpers import format_number
        extra_text_summaries = [f"Power: {format_number(power)}\n95% CI: ({format_number(average_ci[0])}, {format_number(average_ci[1])})"]

        return df, figures, extra_text_summaries

    return None
