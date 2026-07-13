"""
Renders the CRM simulation results view in the Streamlit dashboard.


Random Seed Strategy: {crm_view_seed_strategy}
"""

import streamlit as st

from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


from clintrials.core.registry import PROTOCOL_REGISTRY

def crm_preview_sims(target_tox, cohort_size, max_size):
    """
    Generate preview simulations for the CRM model.
    """
    from clintrials.dosefinding.crm import CRM
    from clintrials.dosefinding import simulate_dose_finding_trial
    
    crm = CRM(
        prior=[0.05, 0.1, 0.2, 0.3, 0.4], 
        target=target_tox, 
        first_dose=1, 
        max_size=max_size
    )
    scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4, 0.5)]
    sims = []
    for true_tox in scenarios:
        for _ in range(20):
            report = simulate_dose_finding_trial(crm, true_toxicities=true_tox, cohort_size=cohort_size)
            report["true_tox"] = true_tox
            sims.append(report)
    return sims

@PROTOCOL_REGISTRY.register("CRM", preview_func=crm_preview_sims)
@dashboard_view(title="CRM Simulation Results", model_name="CRM", file_prefix="crm_simulations")
def render(sims):
    """Renders the CRM simulation results view."""
    st.sidebar.header("Define Parameter Space for CRM")
    param_space_config = {
        "true_tox": [(0.05, 0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4, 0.5)]
    }
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.crm import CRM
    func_map = CRM.get_summary_functions()

    summary_df = extract_sim_data(sims, ps, func_map, return_type="dataframe")
    
    figures = []
    if not summary_df.empty and "recommended_dose_prob" in summary_df.columns:
        import clintrials.visualization as viz
        fig = viz.plot_crm_simulation_recommendation(
            summary_df,
            high_contrast=False
        )
        figures.append(("Dose Recommendation Probability", fig))
    else:
        st.warning(
            "Could not generate Dose Recommendation Probability plot. Check simulation data and summary output."
        )

    return summary_df, figures
