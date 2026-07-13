"""
Renders the EffTox simulation results view in the Streamlit dashboard.


Random Seed Strategy: {efftox_view_seed_strategy}
"""

import streamlit as st

from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


from clintrials.core.registry import PROTOCOL_REGISTRY

def efftox_preview_sims(target_tox, cohort_size, max_size):
    from clintrials.dosefinding.efftox import EffTox, LpNormCurve
    from clintrials.dosefinding.efficacytoxicity import simulate_trial
    
    real_doses = [1.0, 2.0, 3.0, 4.0, 5.0]
    prior_tox_probs = [0.05, 0.1, 0.2, 0.3, 0.4]
    prior_eff_probs = [0.2, 0.4, 0.6, 0.7, 0.8]
    
    metric = LpNormCurve(0.2, 0.4, 0.5, 0.2)
    trial = EffTox(
        real_doses=real_doses,
        prior_tox_probs=prior_tox_probs,
        prior_eff_probs=prior_eff_probs,
        tox_cutoff=0.4,
        eff_cutoff=0.2,
        tox_certainty=0.8,
        eff_certainty=0.8,
        metric=metric,
        max_size=max_size,
    )
    
    tox_scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4)]
    eff_scenarios = [(0.2, 0.3, 0.4, 0.5, 0.6)]
    
    sims = []
    for t_tox in tox_scenarios:
        for t_eff in eff_scenarios:
            for _ in range(10):
                report = simulate_trial(trial, true_toxicities=t_tox, true_efficacies=t_eff, cohort_size=cohort_size)
                report["true_prob_tox"] = t_tox
                report["true_prob_eff"] = t_eff
                sims.append(report)
    return sims

@PROTOCOL_REGISTRY.register("EffTox", preview_func=efftox_preview_sims)
@dashboard_view(title="EffTox Simulation Results", model_name="EffTox", file_prefix="efftox_simulations")
def render(sims):
    """Renders the EffTox simulation results view."""
    param_space_config = {
        "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
        "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
    }
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.efftox import EffTox
    func_map = EffTox.get_summary_functions()

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
