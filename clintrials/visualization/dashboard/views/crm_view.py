"""
Renders the CRM simulation results view in the Streamlit dashboard.


Random Seed Strategy: {crm_view_seed_strategy}
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
        summary_df = extract_sim_data(sims, ps, func_map, to_pandas=True)

        st.subheader("Simulation Summary")
        st.write(summary_df)

        text_summaries = []

        st.header("Operating Characteristics")

        if not summary_df.empty and "recommended_dose_prob" in summary_df.columns:
            st.subheader("Dose Recommendation Probability")
            import clintrials.visualization as viz

            fig = viz.plot_crm_simulation_recommendation(
                summary_df,
                high_contrast=getattr(st, "session_state", {}).get("accessibility_mode", False)
            )
            meta = getattr(
                getattr(fig, "layout", None), "meta", "No data summary available."
            )
            text_summaries.append(meta)

            from clintrials.visualization.dashboard.factory import render_accessible_chart
            render_accessible_chart(st, fig)
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

        viz_provider = get_visualization_provider()
        pdf_data = viz_provider.generate_pdf_report(
            summary_df, "CRM", text_summaries=text_summaries
        ) if viz_provider else None

        if pdf_data is not None:
            getattr(col2, "download_button", lambda *args, **kwargs: None)(
                label="Download PDF",
                data=pdf_data,
                file_name="crm_simulations.pdf",
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
