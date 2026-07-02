"""
Renders the WATU simulation results view in the Streamlit dashboard.
"""

import streamlit as st

from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


@dashboard_view(title="WATU Simulation Results", model_name="WATU", file_prefix="watu_simulations")
def render(sims):
    """Renders the WATU simulation results view."""
    param_space_config = {
        "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
        "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
    }
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.watu import WATU
    func_map = WATU.get_summary_functions()

    var_map = {
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    summary_df = extract_sim_data(sims, ps, func_map, var_map=var_map, return_type="dataframe")

    figures = []
    if not summary_df.empty:
        if "recommended_dose_prob" in summary_df.columns:
            import clintrials.visualization as viz
            fig_rec = viz.plot_efftox_simulation_recommendation(
                summary_df,
                high_contrast=getattr(st, "session_state", {}).get("accessibility_mode", False)
            )
            figures.append(("Dose Recommendation Probability", fig_rec))
    else:
        st.warning("Summary dataframe is empty. Cannot generate plots.")

    return summary_df, figures
