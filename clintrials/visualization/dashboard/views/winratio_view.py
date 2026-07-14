"""
Renders the Win Ratio simulation view in the Streamlit dashboard.


Random Seed Strategy: {winratio_view_seed_strategy}
"""

import pandas as pd
import streamlit as st

from clintrials.winratio.main import WinRatioTrial
from clintrials.core.schema import WinRatioSchema
from clintrials.visualization.dashboard.factory import create_widget, render_metric
from clintrials.visualization.dashboard.views.framework import dashboard_view


from clintrials.core.registry import PROTOCOL_REGISTRY

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
        with st.spinner("Running simulation..."):
            trial = WinRatioTrial(**kwargs)
            trial.update()
            power = trial.power
            average_ci = trial.average_ci
        st.success("Simulation complete")
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
        
        extra_text_summaries = [f"Power: {power:.4f}\n95% CI: ({average_ci[0]:.4f}, {average_ci[1]:.4f})"]

        return df, figures, extra_text_summaries

    return None
