"""Renders the CRM simulation results view in the Streamlit dashboard.

Random Seed Strategy: {crm_view_seed_strategy}
"""

import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


def crm_preview_sims(target_tox, cohort_size, max_size):
    """Generate preview simulations for the CRM model.
    """
    from clintrials.dosefinding import simulate_dose_finding_trial
    from clintrials.dosefinding.crm import CRM
    from clintrials.core.simulation import sim_parameter_space
    from clintrials.utils import ParameterSpace

    crm = CRM(
        prior=[0.05, 0.1, 0.2, 0.3, 0.4],
        target=target_tox,
        first_dose=1,
        max_size=max_size
    )
    
    ps = ParameterSpace()
    ps.add("true_tox", [(0.05, 0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4, 0.5)])

    def wrapped_sim_func(true_tox):
        report = simulate_dose_finding_trial(crm, true_toxicities=true_tox, cohort_size=cohort_size)
        report["true_tox"] = true_tox
        return report

    sims = sim_parameter_space(wrapped_sim_func, ps, n1=20)
    return sims

@PROTOCOL_REGISTRY.register("CRM", preview_func=crm_preview_sims)
@dashboard_view(title="CRM Simulation Results", model_name="CRM", file_prefix="crm_simulations")
def render(sims):
    """Renders the CRM simulation results view."""
    st.sidebar.header("Trial Parameters")
    param_space_config = {
        "true_tox": [(0.05, 0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4, 0.5)]
    }
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)

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
