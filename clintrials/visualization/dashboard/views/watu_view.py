"""
Renders the WATU simulation results view in the Streamlit dashboard.
"""

import json

import pandas as pd
import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.core.simulation import summarise_sims
from clintrials.utils import ParameterSpace
from clintrials.core.viz_interface import get_visualization_provider


def render(sims):
    """Renders the WATU simulation results view."""
    st.header("WATU Simulation Results")

    param_space_config = {
        "true_prob_tox": [[0.05, 0.1, 0.2, 0.3, 0.4]],
        "true_prob_eff": [[0.2, 0.3, 0.4, 0.5, 0.6]],
    }
    ps = ParameterSpace(param_space_config)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.watu import WATU

    func_map = WATU.get_summary_functions()

    var_map = {
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    try:
        summary_df = summarise_sims(sims, ps, func_map, var_map=var_map, to_pandas=True)

        st.subheader("Simulation Summary")
        st.write(summary_df)

        st.header("Operating Characteristics")

        text_summaries = []

        if not summary_df.empty:
            if "recommended_dose_prob" in summary_df.columns:
                st.subheader("Dose Recommendation Probability")
                import clintrials.visualization as viz

                fig_rec = viz.plot_efftox_simulation_recommendation(summary_df)
                meta_rec = getattr(
                    getattr(fig_rec, "layout", None),
                    "meta",
                    "No data summary available.",
                )
                text_summaries.append(meta_rec)

                if getattr(st, "session_state", {}).get("accessibility_mode", False):
                    st.markdown(meta_rec)
                else:
                    st.plotly_chart(fig_rec)
                    with st.expander("Data Summary"):
                        st.markdown(meta_rec)

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
            file_name="watu_simulations.csv",
            mime="text/csv",
        )

        viz_provider = get_visualization_provider()
        pdf_data = viz_provider.generate_pdf_report(
            summary_df, "WATU", text_summaries=text_summaries
        ) if viz_provider else None

        if pdf_data is not None:
            getattr(col2, "download_button", lambda *args, **kwargs: None)(
                label="Download PDF",
                data=pdf_data,
                file_name="watu_simulations.pdf",
                mime="application/pdf",
            )
        else:
            col2.warning("PDF export requires the 'fpdf2' package.")

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")
