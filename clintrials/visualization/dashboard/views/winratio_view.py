"""
Renders the Win Ratio simulation view in the Streamlit dashboard.


Random Seed Strategy: {winratio_view_seed_strategy}
"""

import pandas as pd
import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.winratio.main import WinRatioTrial
from clintrials.core.schema import WinRatioSchema
from clintrials.visualization.dashboard.factory import create_widget
from clintrials.core.report import generate_pdf_report


def render() -> None:
    """Render the Win Ratio simulation interface."""
    st.header("Win Ratio Simulation")

    st.sidebar.header("Simulation Parameters")

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
        st.write(f"Power of the test: {power:.4f}")
        st.write(
            "Average 95% Confidence Interval: "
            f"({average_ci[0]:.4f}, {average_ci[1]:.4f})"
        )

        # Create simple DataFrame for export
        results_dict = kwargs.copy()
        results_dict["power"] = power
        results_dict["ci_lower"] = average_ci[0]
        results_dict["ci_upper"] = average_ci[1]

        df = pd.DataFrame([results_dict])

        st.header("Export Results")
        if not hasattr(st, "columns"):
            st.columns = lambda x: (st, st)
        col1, col2 = st.columns(2)

        csv_data = df.to_csv(index=False)
        getattr(col1, "download_button", lambda *args, **kwargs: None)(
            label="Download CSV",
            data=csv_data,
            file_name="winratio_simulation.csv",
            mime="text/csv",
        )

        pdf_data = generate_pdf_report(
            df,
            "Win Ratio",
            text_summaries=[
                f"Power: {power:.4f}\n95% CI: ({average_ci[0]:.4f}, {average_ci[1]:.4f})"
            ],
        )
        getattr(col2, "download_button", lambda *args, **kwargs: None)(
            label="Download PDF",
            data=pdf_data,
            file_name="winratio_simulation.pdf",
            mime="application/pdf",
        )


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY
    __doc__ = __doc__.format(**REGISTRY)
