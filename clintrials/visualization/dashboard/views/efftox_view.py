"""Renders the EffTox simulation results view in the Streamlit dashboard.

Random Seed Strategy: {efftox_view_seed_strategy}
"""

import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.core.simulation import extract_sim_data
from clintrials.visualization.dashboard.views.framework import dashboard_view


def efftox_preview_sims(target_tox, cohort_size, max_size):
    """Generate preview simulations for the EffTox model.
    """
    from clintrials.core.simulation import run_bivariate_simulations
    from clintrials.dosefinding.efftox import EffTox, LpNormCurve

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

    return run_bivariate_simulations(trial, tox_scenarios, eff_scenarios, cohort_size, n_replicates=10)

@PROTOCOL_REGISTRY.register("EffTox", preview_func=efftox_preview_sims)
@dashboard_view(title="EffTox Simulation Results", model_name="EffTox", file_prefix="efftox_simulations", param_space_config={
    "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
    "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
})
def render(sims, ps):
    """Renders the EffTox simulation results view."""
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
            fig_rec = viz.plot_bivariate_simulation_recommendation(
                summary_df,
                high_contrast=False
            )
            figures.append(("Dose Recommendation Probability", fig_rec))

        if (
            "prob_accept_tox" in summary_df.columns
            and "prob_accept_eff" in summary_df.columns
        ):
            import clintrials.visualization as viz
            fig_accept = viz.plot_efftox_simulation_acceptability(
                summary_df,
                high_contrast=False
            )
            figures.append(("Acceptability Probabilities", fig_accept))
    else:
        st.warning("Summary dataframe is empty. Cannot generate plots.")

    return summary_df, figures
