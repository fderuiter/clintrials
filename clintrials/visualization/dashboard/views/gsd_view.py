"""Renders the Group Sequential Design simulation view in the Streamlit dashboard."""

import pandas as pd
import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.phase3.gsd import (
    GroupSequentialDesign,
    spending_function_obrien_fleming,
    spending_function_pocock,
)
from clintrials.visualization.dashboard.factory import create_widget
from clintrials.visualization.dashboard.views.framework import dashboard_view


@PROTOCOL_REGISTRY.register("Group Sequential Design")
@dashboard_view(
    title="Group Sequential Design Simulation",
    model_name="Group Sequential Design",
    file_prefix="gsd_simulation",
    csv_index=False,
    skip_summary_table=True
)
def render() -> None:
    """Render the Group Sequential Design simulation interface."""
    st.sidebar.header("Trial Parameters")

    k = create_widget(
        st,
        "number_input",
        "gsd_k",
        "Number of analyses (looks)",
        min_value=1,
        max_value=10,
        value=3,
    )

    alpha = create_widget(
        st,
        "number_input",
        "gsd_alpha",
        "Significance Level (Alpha)",
        min_value=0.001,
        max_value=0.999,
        value=0.025,
        step=0.005,
    )

    sfu_name = create_widget(
        st,
        "selectbox",
        "gsd_sfu",
        "Spending Function",
        ("O'Brien-Fleming", "Pocock"),
        index=0
    )

    n_sims = create_widget(
        st,
        "number_input",
        "gsd_n_sims",
        "Number of Simulations",
        min_value=10,
        max_value=10000,
        value=1000,
        step=100,
    )

    theta = create_widget(
        st,
        "number_input",
        "gsd_theta",
        "True Effect Size (Theta)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )

    if create_widget(st, "button", "run_simulation_button", "Run Simulation"):
        from clintrials.visualization.dashboard.utils import announce_status_locally
        announce_status_locally("Simulation in progress", key="gsd-start")
        try:
            with st.spinner("Running simulation..."):
                sfu = spending_function_pocock if sfu_name == "Pocock" else spending_function_obrien_fleming
                gsd = GroupSequentialDesign(k=k, alpha=alpha, sfu=sfu)
                # Ensure we use .simulate() which was on the whitelist
                sims = gsd.simulate(n_sims=n_sims, theta=theta)
            announce_status_locally("Simulation completed", key="gsd-complete")
            st.success("Simulation complete")
        except Exception as e:
            announce_status_locally("Simulation failed", key="gsd-fail")
            raise e

        st.subheader("Results")

        # Compute summary
        rejected = [sim.get("Rejected", False) for sim in sims]
        power = sum(rejected) / len(rejected) if rejected else 0.0

        st.write(f"Empirical Power / Type I Error: {power:.4f}")

        results_dict = {
            "k": k,
            "alpha": alpha,
            "sfu": sfu_name,
            "n_sims": n_sims,
            "theta": theta,
            "power": power
        }

        from collections import Counter
        stop_stages = [sim.get("Stage", k) for sim in sims]
        stage_counts = Counter(stop_stages)

        stages = list(range(1, k + 1))
        counts = [stage_counts.get(s, 0) for s in stages]

        plot_df = pd.DataFrame({
            "Stage": stages,
            "Count": counts,
            "Outcome": ["Stop" for _ in stages]
        })

        import clintrials.visualization as viz
        fig = viz.create_bar_chart(
            plot_df,
            x="Stage",
            y="Count",
            color="Outcome",
            title="Trial Progression (Stop Stages)"
        )
        figures = [("Trial Progression (Stop Stages)", fig)]

        return pd.DataFrame([results_dict]), figures, []

    return None
