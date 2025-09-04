import streamlit as st

from clintrials.winratio import run_simulation


def render() -> None:
    """Render the Win Ratio simulation interface."""
    st.header("Win Ratio Simulation")

    st.sidebar.header("Simulation Parameters")
    num_subjects_A = st.sidebar.number_input(
        "Number of subjects in Group A", min_value=1, value=100
    )
    num_subjects_B = st.sidebar.number_input(
        "Number of subjects in Group B", min_value=1, value=50
    )
    num_simulations = st.sidebar.number_input(
        "Number of simulations", min_value=1, value=1000
    )
    p_y1_A = st.sidebar.number_input(
        "Probability of y1=1 for Group A", min_value=0.0, max_value=1.0, value=0.50
    )
    p_y1_B = st.sidebar.number_input(
        "Probability of y1=1 for Group B", min_value=0.0, max_value=1.0, value=0.50
    )
    p_y2_A = st.sidebar.number_input(
        "Probability of y2=1 for Group A", min_value=0.0, max_value=1.0, value=0.75
    )
    p_y2_B = st.sidebar.number_input(
        "Probability of y2=1 for Group B", min_value=0.0, max_value=1.0, value=0.25
    )
    p_y3_A = st.sidebar.number_input(
        "Probability of y3=1 for Group A", min_value=0.0, max_value=1.0, value=0.43
    )
    p_y3_B = st.sidebar.number_input(
        "Probability of y3=1 for Group B", min_value=0.0, max_value=1.0, value=0.27
    )
    significance_level = st.sidebar.number_input(
        "Significance level", min_value=0.0, max_value=1.0, value=0.05
    )

    if st.sidebar.button("Run Simulation"):
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
