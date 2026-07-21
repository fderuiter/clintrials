"""Renders the WATU simulation results view in the Streamlit dashboard."""

import streamlit as st

from clintrials.dosefinding.watu import WATU
from clintrials.visualization.dashboard.views.framework import BaseSimulationView


class WATUView(BaseSimulationView):
    """View class for the Wages and Tait (WATU) model."""

    model_name = "WATU"
    title = "WATU Simulation Results"
    file_prefix = "watu_simulations"
    model_class = WATU
    param_space_config = {
        "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
        "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
    }
    var_map = {
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    @classmethod
    def preview_sims(cls, target_tox, cohort_size, max_size):
        """Generate preview simulations for the WATU model."""
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

        watu = WATU(
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

    @classmethod
    def build_figures(cls, summary_df):
        """Generate visualization plots for the WATU summary dataframe."""
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

        return figures
