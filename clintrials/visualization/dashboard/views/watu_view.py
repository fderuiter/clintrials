"""Renders the WATU simulation results view in the Streamlit dashboard."""

import streamlit as st

from clintrials.dosefinding.watu import WATU
from clintrials.visualization.dashboard.views.framework import BaseSimulationView


class WATUView(BaseSimulationView):  # type: ignore
    """View class for the Wages and Tait (WATU) model."""

    model_name = "WATU"
    title = "WATU Simulation Results"
    file_prefix = "watu_simulations"
    model_class = WATU  # type: ignore
    param_space_config = {
        "true_prob_tox": [(0.05, 0.1, 0.2, 0.3, 0.4)],
        "true_prob_eff": [(0.2, 0.3, 0.4, 0.5, 0.6)],
    }
    var_map = {  # type: ignore
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    @classmethod
    def preview_sims(cls, target_tox, cohort_size, max_size):  # type: ignore
        """Generate preview simulations for the WATU model."""
        from clintrials.core.simulation import run_bivariate_simulations
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
        return run_bivariate_simulations(watu, tox_scenarios, eff_scenarios, cohort_size, n_replicates=10)  # type: ignore

    @classmethod
    def build_figures(cls, summary_df):  # type: ignore
        """Generate visualization plots for the WATU summary dataframe."""
        figures = []
        if not summary_df.empty:
            if "recommended_dose_prob" in summary_df.columns:
                import clintrials.visualization as viz

                fig_rec = viz.plot_bivariate_simulation_recommendation(  # type: ignore
                    summary_df,
                    high_contrast=False
                )
                figures.append(("Dose Recommendation Probability", fig_rec))
        else:
            st.warning("Summary dataframe is empty. Cannot generate plots.")

        return figures
