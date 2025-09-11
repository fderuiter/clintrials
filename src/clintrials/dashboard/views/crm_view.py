import json

import pandas as pd
import plotly.express as px
import streamlit as st

from clintrials.core.simulation import summarise_sims
from clintrials.utils import ParameterSpace


def render(sims):
    """Renders the CRM simulation results view.

    Args:
        sims: A list of simulation results.
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

    # Define summary functions for CRM
    func_map = {
        "N": lambda s, p: len(s),
        "recommended_dose_prob": lambda s, p: pd.Series(
            [x["recommended_dose"] for x in s]
        )
        .value_counts(normalize=True)
        .sort_index(),
    }

    try:
        summary_df = summarise_sims(sims, ps, func_map, to_pandas=True)

        st.subheader("Simulation Summary")
        st.write(summary_df)

        # Plotting
        st.header("Operating Characteristics")

        if not summary_df.empty and "recommended_dose_prob" in summary_df.columns:
            st.subheader("Dose Recommendation Probability")

            # The 'recommended_dose_prob' column contains dictionaries. We need to expand it into a DataFrame.
            rec_dose_df = summary_df["recommended_dose_prob"].apply(pd.Series).fillna(0)

            # We need to melt the DataFrame to plot it with plotly express
            rec_dose_df_melted = rec_dose_df.reset_index().melt(
                id_vars=[col for col in rec_dose_df.index.names],
                var_name="Dose Level",
                value_name="Probability",
            )

            # Create an interactive bar chart
            fig = px.bar(
                rec_dose_df_melted,
                x="true_tox",
                y="Probability",
                color="Dose Level",
                barmode="group",
                labels={
                    "true_tox": "True Toxicity Scenario",
                    "Probability": "Recommendation Probability",
                },
                title="Dose Recommendation Probabilities by Scenario",
            )
            st.plotly_chart(fig)

        else:
            st.warning(
                "Could not generate Dose Recommendation Probability plot. Check simulation data and summary output."
            )

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")
