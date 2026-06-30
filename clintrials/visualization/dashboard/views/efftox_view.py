"""
Renders the EffTox simulation results view in the Streamlit dashboard.
"""

import json

import pandas as pd
import streamlit as st

if not hasattr(st, "fragment"):
    st.fragment = lambda func: func

from clintrials.core.simulation import summarise_sims
from clintrials.utils import ParameterSpace


def render(sims):
    """Renders the EffTox simulation results view.

    This function displays the results of EffTox simulations, including a
    summary table and plots of dose recommendation and acceptability
    probabilities.

    Args:
        sims (list[dict]): A list of simulation results.
    """
    st.header("EffTox Simulation Results")

    # Example parameter space for EffTox
    param_space_config = {
        "true_prob_tox": [[0.05, 0.1, 0.2, 0.3, 0.4]],
        "true_prob_eff": [[0.2, 0.3, 0.4, 0.5, 0.6]],
    }
    ps = ParameterSpace(param_space_config)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    from clintrials.dosefinding.efftox import EffTox
    func_map = EffTox.get_summary_functions()

    # Need to map the parameter space names to the names in the simulation file
    var_map = {
        "true_prob_tox": "true_prob_tox",
        "true_prob_eff": "true_prob_eff",
    }

    try:
        summary_df = summarise_sims(sims, ps, func_map, var_map=var_map, to_pandas=True)

        st.subheader("Simulation Summary")
        st.write(summary_df)

        # Plotting
        st.header("Operating Characteristics")

        if not summary_df.empty:
            # Dose Recommendation Probability
            if "recommended_dose_prob" in summary_df.columns:
                st.subheader("Dose Recommendation Probability")
                import clintrials.visualization as viz

                result_rec = viz.plot_efftox_simulation_recommendation(summary_df)
                st.plotly_chart(result_rec.chart)
                with st.expander("Data Summary"):
                    if result_rec.metadata is not None:
                        st.dataframe(result_rec.metadata)
                    else:
                        st.write("No data summary available.")

            # Acceptability Probabilities
            if (
                "prob_accept_tox" in summary_df.columns
                and "prob_accept_eff" in summary_df.columns
            ):
                st.subheader("Acceptability Probabilities")
                import clintrials.visualization as viz

                result_accept = viz.plot_efftox_simulation_acceptability(summary_df)
                st.plotly_chart(result_accept.chart)
                with st.expander("Data Summary"):
                    if result_accept.metadata is not None:
                        st.dataframe(result_accept.metadata)
                    else:
                        st.write("No data summary available.")

        else:
            st.warning("Summary dataframe is empty. Cannot generate plots.")

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")
