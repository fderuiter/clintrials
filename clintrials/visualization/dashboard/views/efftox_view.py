"""
Renders the EffTox simulation results view in the Streamlit dashboard.


Random Seed Strategy: {efftox_view_seed_strategy}
"""

import json
import pandas as pd
import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.core.simulation import extract_sim_data
from clintrials.utils import ParameterSpace
from clintrials.core.viz_interface import get_visualization_provider


def render(sims):
    """Renders the EffTox simulation results view."""
    st.header("EffTox Simulation Results")

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

    try:
        summary_df = extract_sim_data(sims, ps, func_map, var_map=var_map, return_type="dataframe")

        st.subheader("Simulation Summary")
        st.write(summary_df)

        st.header("Operating Characteristics")

        text_summaries = []

        if not summary_df.empty:
            if "recommended_dose_prob" in summary_df.columns:
                st.subheader("Dose Recommendation Probability")
                import clintrials.visualization as viz

                fig_rec = viz.plot_efftox_simulation_recommendation(
                    summary_df,
                    high_contrast=getattr(st, "session_state", {}).get("accessibility_mode", False)
                )
                meta_rec = getattr(
                    getattr(fig_rec, "layout", None),
                    "meta",
                    "No data summary available.",
                )
                text_summaries.append(meta_rec)

                from clintrials.visualization.dashboard.factory import render_accessible_chart
                render_accessible_chart(st, fig_rec)

            if (
                "prob_accept_tox" in summary_df.columns
                and "prob_accept_eff" in summary_df.columns
            ):
                st.subheader("Acceptability Probabilities")
                import clintrials.visualization as viz

                fig_accept = viz.plot_efftox_simulation_acceptability(
                    summary_df,
                    high_contrast=getattr(st, "session_state", {}).get("accessibility_mode", False)
                )
                meta_accept = getattr(
                    getattr(fig_accept, "layout", None),
                    "meta",
                    "No data summary available.",
                )
                text_summaries.append(meta_accept)

                from clintrials.visualization.dashboard.factory import render_accessible_chart
                render_accessible_chart(st, fig_accept)
        else:
            st.warning("Summary dataframe is empty. Cannot generate plots.")

        st.header("Export Results")
        if not hasattr(st, "columns"):
            st.columns = lambda x: (st, st)
        col1, col2 = st.columns(2)

        csv_data = summary_df.to_csv()
        getattr(col1, "download_button", lambda *args, **kwargs: None)(
            label="Download CSV",
            data=csv_data,
            file_name="efftox_simulations.csv",
            mime="text/csv",
        )

        viz_provider = get_visualization_provider()
        pdf_data = viz_provider.generate_pdf_report(
            summary_df, "EffTox", text_summaries=text_summaries
        ) if viz_provider else None

        if pdf_data is not None:
            getattr(col2, "download_button", lambda *args, **kwargs: None)(
                label="Download PDF",
                data=pdf_data,
                file_name="efftox_simulations.pdf",
                mime="application/pdf",
            )
        else:
            col2.warning("PDF export requires the 'fpdf2' package.")

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY
    __doc__ = __doc__.format(**REGISTRY)
