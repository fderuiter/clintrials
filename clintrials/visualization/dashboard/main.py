"""
Main entry point for the Streamlit dashboard.


Random Seed Strategy: {main_seed_strategy}
"""

import json

import streamlit as st

from clintrials.visualization.dashboard.views import (
    crm_view,
    efftox_view,
    winratio_view,
)
from clintrials.visualization.dashboard.factory import create_widget


def main():
    """Sets up the Streamlit dashboard and renders the appropriate view.

    This function creates the main layout of the dashboard, including the
    sidebar for selecting the trial design and uploading simulation results.
    It then calls the appropriate render function based on the user's
    selection.
    """
    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Accessibility")
    if not hasattr(st, "session_state"):
        st.session_state = {}
    if "accessibility_mode" not in st.session_state:
        st.session_state["accessibility_mode"] = False

    toggle_fn = getattr(st.sidebar, "toggle", lambda *args, **kwargs: False)
    st.session_state["accessibility_mode"] = toggle_fn(
        "Accessibility Mode",
        value=st.session_state.get("accessibility_mode", False),
        help="Enable high-fidelity text alternatives for screen readers.",
    )

    st.sidebar.header("Select Trial Design")
    design_type = create_widget(
        st,
        "selectbox",
        "design_type",
        "Choose the type of trial design for your simulation results:",
        ("CRM", "EffTox", "Win Ratio"),
    )

    if design_type == "Win Ratio":
        winratio_view.render()
    else:
        st.sidebar.header("Upload Simulation Results")
        uploaded_file = create_widget(
            st,
            "file_uploader",
            "uploaded_file",
            "Upload a JSON file with simulation results",
            type=["json"],
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


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY
    __doc__ = __doc__.format(**REGISTRY)
