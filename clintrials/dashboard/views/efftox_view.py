import json

import pandas as pd
import plotly.express as px
import streamlit as st

from clintrials.simulation import summarise_sims
from clintrials.util import ParameterSpace


def render(sims):
    st.header("EffTox Simulation Results")

    # Example parameter space for EffTox
    param_space_config = {
        "true_prob_tox": [[0.05, 0.1, 0.2, 0.3, 0.4]],
        "true_prob_eff": [[0.2, 0.3, 0.4, 0.5, 0.6]],
    }
    ps = ParameterSpace(param_space_config)

    st.sidebar.write("Parameter space for summarization:")
    st.sidebar.json(param_space_config)

    # Define summary functions for EffTox
    func_map = {
        "N": lambda s, p: len(s),
        "recommended_dose_prob": lambda s, p: pd.Series(
            [x.get("recommended_dose") for x in s]
        )
        .value_counts(normalize=True)
        .sort_index(),
        "prob_accept_tox": lambda s, p: pd.Series(
            [x.get("prob_accept_tox", 0) > 0.5 for x in s]
        ).mean(),
        "prob_accept_eff": lambda s, p: pd.Series(
            [x.get("prob_accept_eff", 0) > 0.5 for x in s]
        ).mean(),
    }

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
                rec_dose_df = (
                    summary_df["recommended_dose_prob"].apply(pd.Series).fillna(0)
                )
                rec_dose_df_melted = rec_dose_df.reset_index().melt(
                    id_vars=[col for col in rec_dose_df.index.names],
                    var_name="Dose Level",
                    value_name="Probability",
                )
                fig_rec = px.bar(
                    rec_dose_df_melted,
                    x=["true_prob_tox", "true_prob_eff"],
                    y="Probability",
                    color="Dose Level",
                    barmode="group",
                    title="Dose Recommendation Probabilities",
                )
                st.plotly_chart(fig_rec)

            # Acceptability Probabilities
            if (
                "prob_accept_tox" in summary_df.columns
                and "prob_accept_eff" in summary_df.columns
            ):
                st.subheader("Acceptability Probabilities")
                accept_df = summary_df[
                    ["prob_accept_tox", "prob_accept_eff"]
                ].reset_index()
                accept_df_melted = accept_df.melt(
                    id_vars=["true_prob_tox", "true_prob_eff"],
                    var_name="Probability Type",
                    value_name="Probability",
                )
                fig_accept = px.line(
                    accept_df_melted,
                    x="true_prob_tox",  # This might need to be more clever
                    y="Probability",
                    color="Probability Type",
                    title="Probability of Acceptable Efficacy and Toxicity",
                )
                st.plotly_chart(fig_accept)

        else:
            st.warning("Summary dataframe is empty. Cannot generate plots.")

    except Exception as e:
        st.error(f"An error occurred during summarization or plotting: {e}")
