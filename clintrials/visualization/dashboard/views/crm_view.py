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


def render(sims):
    """Renders the CRM simulation results view.

    This function displays the results of CRM simulations, including a
    summary table and a plot of dose recommendation probabilities.

    Args:
        sims (list[dict]): A list of simulation results.
    """
    st.header("CRM Simulation Results")

    # For this proof-of-concept, we'll make some assumptions about the data.
    st.sidebar.header("Define Parameter Space for CRM")

    # Example parameter space for CRM - this should be adapted based on expected sim structure
    # This assumes the simulations were run over a parameter space of true toxicity probabilities.
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

        # Plotting
        st.header("Operating Characteristics")

        if not summary_df.empty and "recommended_dose_prob" in summary_df.columns:
            st.subheader("Dose Recommendation Probability")
            import clintrials.visualization as viz

            fig = viz.plot_crm_simulation_recommendation(summary_df)
            st.plotly_chart(fig)
            with st.expander("Data Summary"):
                st.markdown(
                    getattr(
                        getattr(fig, "layout", None),
                        "meta",
                        "No data summary available.",
                    )
                )

        else:
            st.warning(
                "Could not generate Dose Recommendation Probability plot. Check simulation data and summary output."
            )

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")
