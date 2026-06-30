"""
Renders the CRM simulation results view in the Streamlit dashboard.
"""

import json
import pandas as pd
import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.core.simulation import summarise_sims
from clintrials.utils import ParameterSpace
from clintrials.core.report import generate_pdf_report


def render(sims):
    """Renders the CRM simulation results view."""
    st.header("CRM Simulation Results")

    st.sidebar.header("Define Parameter Space for CRM")
    param_space_config = {
        "true_tox": [[0.05, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.5]]
    }
    ps = ParameterSpace(param_space_config)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.crm import CRM

    func_map = CRM.get_summary_functions()

    try:
        summary_df = summarise_sims(sims, ps, func_map, to_pandas=True)

        st.subheader("Simulation Summary")
        st.write(summary_df)

        text_summaries = []

        st.header("Operating Characteristics")

        if not summary_df.empty and "recommended_dose_prob" in summary_df.columns:
            st.subheader("Dose Recommendation Probability")
            import clintrials.visualization as viz

            fig = viz.plot_crm_simulation_recommendation(summary_df)
            meta = getattr(
                getattr(fig, "layout", None), "meta", "No data summary available."
            )
            text_summaries.append(meta)

            if getattr(st, "session_state", {}).get("accessibility_mode", False):
                st.markdown(meta)
            else:
                st.plotly_chart(fig)
                with st.expander("Data Summary"):
                    st.markdown(meta)
        else:
            st.warning(
                "Could not generate Dose Recommendation Probability plot. Check simulation data and summary output."
            )

        st.header("Export Results")
        if not hasattr(st, "columns"):
            st.columns = lambda x: (st, st)
        col1, col2 = st.columns(2)

        csv_data = summary_df.to_csv()
        getattr(col1, "download_button", lambda *args, **kwargs: None)(
            label="Download CSV",
            data=csv_data,
            file_name="crm_simulations.csv",
            mime="text/csv",
        )

        pdf_data = generate_pdf_report(summary_df, "CRM", text_summaries=text_summaries)
        getattr(col2, "download_button", lambda *args, **kwargs: None)(
            label="Download PDF",
            data=pdf_data,
            file_name="crm_simulations.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")
