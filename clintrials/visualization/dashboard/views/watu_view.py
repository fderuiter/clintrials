from __future__ import annotations
"""Renders the WATU simulation results view in the Streamlit dashboard.
"""

import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.visualization.dashboard.views.framework import dashboard_view


def watu_preview_sims(target_tox, cohort_size, max_size):  # type: ignore
    """Generate preview simulations for the WATU model.
    """
    from clintrials.dosefinding.efficacytoxicity import simulate_trial
    from clintrials.dosefinding.efftox import LpNormCurve
    from clintrials.dosefinding.watu import WATU
    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20],
        [0.50, 0.60, 0.50, 0.40, 0.30],
        [0.40, 0.50, 0.60, 0.50, 0.40],
        [0.30, 0.40, 0.50, 0.60, 0.50],
        [0.20, 0.30, 0.40, 0.50, 0.60],
    ]
    tox_prior = [0.05, 0.1, 0.2, 0.3, 0.4]
    metric = LpNormCurve(0.2, 0.4, 0.5, 0.2)

    watu = WATU(  # type: ignore
        skeletons=skeletons,
        prior_tox_probs=tox_prior,
        tox_target=target_tox,
        tox_limit=0.4,
        eff_limit=0.2,
        metric=metric,
        first_dose=1,
        max_size=max_size
    )

    tox_scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4)]
    eff_scenarios = [(0.2, 0.3, 0.4, 0.5, 0.6)]
    sims = []
    for t_tox in tox_scenarios:
        for t_eff in eff_scenarios:
            for _ in range(10):
                report = simulate_trial(watu, true_toxicities=t_tox, true_efficacies=t_eff, cohort_size=cohort_size)
                report["true_prob_tox"] = t_tox
                report["true_prob_eff"] = t_eff
                sims.append(report)
    return sims

@PROTOCOL_REGISTRY.register("WATU", preview_func=watu_preview_sims)
@dashboard_view(title="WATU Simulation Results", model_name="WATU", file_prefix="watu_simulations")
def render(sims):  # type: ignore
    """Renders the WATU simulation results view."""
    st.sidebar.header("Trial Parameters")
    param_space_config = {
        "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
        "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
    }
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)

    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.watu import WATU
    func_map = WATU.get_summary_functions()  # type: ignore

    var_map = {
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    summary_df = extract_sim_data(sims, ps, func_map, var_map=var_map, return_type="dataframe")  # type: ignore

    figures = []
    if not summary_df.empty:
        if "recommended_dose_prob" in summary_df.columns:
            import clintrials.visualization as viz
            fig_rec = viz.plot_efftox_simulation_recommendation(  # type: ignore
                summary_df,
                high_contrast=False
            )
            figures.append(("Dose Recommendation Probability", fig_rec))
    else:
        st.warning("Summary dataframe is empty. Cannot generate plots.")

    return summary_df, figures
