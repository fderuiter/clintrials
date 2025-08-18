import streamlit as st
from clintrials.dashboard.views import crm_view, efftox_view
import json

def main():
    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Select Trial Design")
    design_type = st.sidebar.selectbox(
        "Choose the type of trial design for your simulation results:",
        ("CRM", "EffTox")
    )

    st.sidebar.header("Upload Simulation Results")
    uploaded_file = st.sidebar.file_uploader("Upload a JSON file with simulation results", type=["json"])

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
