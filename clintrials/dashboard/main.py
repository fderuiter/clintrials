import json

import streamlit as st

from clintrials.dashboard.views import crm_view, efftox_view, winratio_view


def main():
    """Sets up the Streamlit dashboard and renders the appropriate view.

    This function creates the main layout of the dashboard, including the
    sidebar for selecting the trial design and uploading simulation results.
    It then calls the appropriate render function based on the user's
    selection.
    """
    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Select Trial Design")
    design_type = st.sidebar.selectbox(
        "Choose the type of trial design for your simulation results:",
        ("CRM", "EffTox", "Win Ratio"),
    )

    if design_type == "Win Ratio":
        winratio_view.render()
    else:
        st.sidebar.header("Upload Simulation Results")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a JSON file with simulation results", type=["json"]
        )

        if uploaded_file is not None:
            string_data = uploaded_file.getvalue().decode("utf-8")
            sims = json.loads(string_data)
            st.sidebar.success(f"Successfully loaded {len(sims)} simulations.")

            if design_type == "CRM":
                crm_view.render(sims)
            elif design_type == "EffTox":
                efftox_view.render(sims)


if __name__ == "__main__":
    main()
