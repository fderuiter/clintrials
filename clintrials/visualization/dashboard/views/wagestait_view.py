"""Renders the Wages & Tait simulation results view in the Streamlit dashboard."""

import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


def wagestait_preview_sims(target_tox, cohort_size, max_size):
    """Generate preview simulations for the Wages & Tait model."""
    from clintrials.dosefinding.efficacytoxicity import simulate_trial
    from clintrials.dosefinding.wagestait import WagesTait
    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20],
        [0.50, 0.60, 0.50, 0.40, 0.30],
        [0.40, 0.50, 0.60, 0.50, 0.40],
        [0.30, 0.40, 0.50, 0.60, 0.50],
        [0.20, 0.30, 0.40, 0.50, 0.60],
    ]
    tox_prior = [0.05, 0.1, 0.2, 0.3, 0.4]

    wt = WagesTait(
        skeletons=skeletons,
        prior_tox_probs=tox_prior,
        tox_target=target_tox,
        tox_limit=0.4,
        eff_limit=0.2,
        first_dose=1,
        max_size=max_size,
        randomisation_stage_size=max_size // 2,
    )

    tox_scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4)]
    eff_scenarios = [(0.2, 0.3, 0.4, 0.5, 0.6)]
    sims = []
    for t_tox in tox_scenarios:
        for t_eff in eff_scenarios:
            for _ in range(10):
                report = simulate_trial(wt, true_toxicities=t_tox, true_efficacies=t_eff, cohort_size=cohort_size)
                report["true_prob_tox"] = t_tox
                report["true_prob_eff"] = t_eff
                sims.append(report)
    return sims

@PROTOCOL_REGISTRY.register("Wages & Tait", preview_func=wagestait_preview_sims)
@dashboard_view(title="Wages & Tait Simulation Results", model_name="Wages & Tait", file_prefix="wagestait_simulations")
def render(sims):
    """Renders the Wages & Tait simulation results view."""
    st.sidebar.header("Trial Parameters")
    param_space_config = {
        "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
        "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
    }
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)

    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.wagestait import WagesTait
    func_map = WagesTait.get_summary_functions()

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
                high_contrast=False
            )
            figures.append(("Dose Recommendation Probability", fig_rec))
    else:
        st.warning("Summary dataframe is empty. Cannot generate plots.")

    return summary_df, figures
