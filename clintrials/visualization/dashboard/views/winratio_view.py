"""
Renders the Win Ratio simulation view in the Streamlit dashboard.
"""

import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.winratio.main import WinRatioTrial
from clintrials.core.schema import WinRatioSchema
from clintrials.visualization.dashboard.factory import create_widget


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
        
        import clintrials.visualization as viz
        result = viz.plot_winratio_simulations(power, average_ci)
        
        st.plotly_chart(result.chart)
        with st.expander("Data Summary"):
            if result.metadata is not None:
                st.dataframe(result.metadata)
            else:
                st.write("No data summary available.")
