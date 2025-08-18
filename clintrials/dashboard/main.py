import streamlit as st
import pandas as pd
import plotly.express as px
import json
from clintrials.simulation import summarise_sims
from clintrials.util import ParameterSpace

def main():
    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Upload Simulation Results")
    uploaded_file = st.sidebar.file_uploader("Upload a JSON file with simulation results", type=["json"])

    if uploaded_file is not None:
        # To read file as string:
        string_data = uploaded_file.getvalue().decode("utf-8")
        sims = json.loads(string_data)
        st.sidebar.success(f"Successfully loaded {len(sims)} simulations.")

        # Display raw data
        if st.sidebar.checkbox("Show raw simulation data"):
            st.subheader("Raw Simulation Data")
            st.write(sims)

        # For this proof-of-concept, we'll make some assumptions about the data.
        # A more robust implementation would infer this from the simulation data itself.
        st.sidebar.header("Define Parameter Space")

        # Example parameter space - this should be adapted based on expected sim structure
        param_space_config = {
            'true_tox': [[0.05, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.5]],
        }
        ps = ParameterSpace(param_space_config)

        st.sidebar.write("Parameter space for summarization:")
        st.sidebar.json(param_space_config)

        # Define summary functions
        func_map = {
            'N': lambda s, p: len(s),
            'recommended_dose': lambda s, p: pd.Series([x['recommended_dose'] for x in s]).value_counts(normalize=True).to_dict()
        }

        try:
            summary_df = summarise_sims(sims, ps, func_map, to_pandas=True)

            st.subheader("Simulation Summary")
            st.write(summary_df)

            # Plotting
            st.header("Operating Characteristics")

            if not summary_df.empty:
                # Example plot: Recommended dose probabilities
                # This needs to be adapted based on the actual structure of summary_df
                st.subheader("Dose Recommendation Probability")

                # We need to unstack the summary to plot it easily
                if 'recommended_dose' in summary_df.columns:
                    rec_dose_df = summary_df['recommended_dose'].apply(pd.Series).fillna(0)
                    st.bar_chart(rec_dose_df)

                else:
                    st.warning("`recommended_dose` not in summary. Cannot generate plot.")

            else:
                st.warning("Summary dataframe is empty. Cannot generate plots.")

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")


if __name__ == "__main__":
    main()
