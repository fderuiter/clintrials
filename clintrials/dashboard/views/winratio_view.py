"""
Renders the Win Ratio simulation view in the Streamlit dashboard.
"""

import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.winratio import run_simulation
from clintrials.dashboard.factory import create_widget


def render() -> None:
    """Render the Win Ratio simulation interface."""
    st.header("Win Ratio Simulation")

    st.sidebar.header("Simulation Parameters")
    num_subjects_A = create_widget(
        st,
        "number_input",
        "num_subjects_A",
        "Number of subjects in Group A",
        min_value=1,
        value=100,
    )
    num_subjects_B = create_widget(
        st,
        "number_input",
        "num_subjects_B",
        "Number of subjects in Group B",
        min_value=1,
        value=50,
    )
    num_simulations = create_widget(
        st,
        "number_input",
        "num_simulations",
        "Number of simulations",
        min_value=1,
        value=1000,
    )
    p_y1_A = create_widget(
        st,
        "number_input",
        "p_y1_A",
        "Probability of y1=1 for Group A",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
    )
    p_y1_B = create_widget(
        st,
        "number_input",
        "p_y1_B",
        "Probability of y1=1 for Group B",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
    )
    p_y2_A = create_widget(
        st,
        "number_input",
        "p_y2_A",
        "Probability of y2=1 for Group A",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
    )
    p_y2_B = create_widget(
        st,
        "number_input",
        "p_y2_B",
        "Probability of y2=1 for Group B",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
    )
    p_y3_A = create_widget(
        st,
        "number_input",
        "p_y3_A",
        "Probability of y3=1 for Group A",
        min_value=0.0,
        max_value=1.0,
        value=0.43,
    )
    p_y3_B = create_widget(
        st,
        "number_input",
        "p_y3_B",
        "Probability of y3=1 for Group B",
        min_value=0.0,
        max_value=1.0,
        value=0.27,
    )
    significance_level = create_widget(
        st,
        "number_input",
        "significance_level",
        "Significance level",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
    )

    if create_widget(st, "button", "run_simulation_button", "Run Simulation"):
        with st.spinner("Running simulation..."):
            power, average_ci = run_simulation(
                num_subjects_A,
                num_subjects_B,
                num_simulations,
                p_y1_A,
                p_y1_B,
                p_y2_A,
                p_y2_B,
                p_y3_A,
                p_y3_B,
                significance_level,
            )
        st.success("Simulation complete")
        st.subheader("Results")
        st.write(f"Power of the test: {power:.4f}")
        st.write(
            "Average 95% Confidence Interval: "
            f"({average_ci[0]:.4f}, {average_ci[1]:.4f})"
        )
