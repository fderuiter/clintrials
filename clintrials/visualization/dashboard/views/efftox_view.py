"""
Renders the EffTox simulation results view in the Streamlit dashboard.


Random Seed Strategy: {efftox_view_seed_strategy}
"""

import json
import pandas as pd
import streamlit as st

from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


@dashboard_view(title="EffTox Simulation Results", model_name="EffTox", file_prefix="efftox_simulations")
def render(sims):
    """Renders the EffTox simulation results view."""
    param_space_config = {
        "true_prob_tox": [[0.05, 0.1, 0.2, 0.3, 0.4]],
        "true_prob_eff": [[0.2, 0.3, 0.4, 0.5, 0.6]],
    }
    ps = ParameterSpace(param_space_config)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.efftox import EffTox
    func_map = EffTox.get_summary_functions()

    var_map = {
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    summary_df = extract_sim_data(sims, ps, func_map, var_map=var_map, to_pandas=True)

    figures = []
    if not summary_df.empty:
        if "recommended_dose_prob" in summary_df.columns:
            import clintrials.visualization as viz
            fig_rec = viz.plot_efftox_simulation_recommendation(
                summary_df,
                high_contrast=getattr(st, "session_state", {}).get("accessibility_mode", False)
            )
            figures.append(("Dose Recommendation Probability", fig_rec))

        if (
            "prob_accept_tox" in summary_df.columns
            and "prob_accept_eff" in summary_df.columns
        ):
            import clintrials.visualization as viz
            fig_accept = viz.plot_efftox_simulation_acceptability(
                summary_df,
                high_contrast=getattr(st, "session_state", {}).get("accessibility_mode", False)
            )
            figures.append(("Acceptability Probabilities", fig_accept))
    else:
        st.warning("Summary dataframe is empty. Cannot generate plots.")

    return summary_df, figures
