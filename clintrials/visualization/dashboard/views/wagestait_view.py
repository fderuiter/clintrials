# SPDX-License-Identifier: MIT

"""Renders the Wages & Tait simulation results view in the Streamlit dashboard."""

from clintrials.dosefinding.wagestait import WagesTait
from clintrials.visualization.dashboard.views.framework import BaseSimulationView


class WagesTaitView(BaseSimulationView):
    """View class for the Wages & Tait trial model."""

    model_name = "Wages & Tait"
    title = "Wages & Tait Simulation Results"
    file_prefix = "wagestait_simulations"
    model_class = WagesTait  # type: ignore
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
        """Generate preview simulations for the Wages & Tait model."""
        from clintrials.core.simulation import run_bivariate_simulations

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
        return run_bivariate_simulations(
            wt, tox_scenarios, eff_scenarios, cohort_size, n_replicates=10
        )

    @classmethod
    def build_figures(cls, summary_df):  # type: ignore
        """Generate visualization plots for the Wages & Tait summary dataframe."""
        import streamlit as st

        figures = []
        if not summary_df.empty:
            if "recommended_dose_prob" in summary_df.columns:
                from clintrials.core.viz_interface import get_visualization_provider

                fig_rec = get_visualization_provider().plot_bivariate_simulation_recommendation(
                    summary_df, high_contrast=False
                )
                figures.append(("Dose Recommendation Probability", fig_rec))
        else:
            st.warning("Summary dataframe is empty. Cannot generate plots.")

        return figures
