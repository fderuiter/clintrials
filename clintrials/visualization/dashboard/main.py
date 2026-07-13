"""
Main entry point for the Streamlit dashboard.


Random Seed Strategy: {main_seed_strategy}
"""

import json

import streamlit as st

from clintrials.visualization.dashboard.factory import create_widget
from clintrials.core.registry import PROTOCOL_REGISTRY

@st.cache_data(show_spinner=False)
def get_preview_sims(design_type, target_tox, cohort_size, max_size):
    """
    Run and cache default preview simulations based on the selected design type and parameters.
    """
    preview_func = PROTOCOL_REGISTRY.get_preview(design_type)
    if preview_func:
        with st.spinner(f"Running Default Preview Simulation for {design_type}..."):
            return preview_func(target_tox, cohort_size, max_size)
    return []

def main():
    """Sets up the Streamlit dashboard and renders the appropriate view."""
    st.sidebar.markdown(
        '<nav aria-label="Settings Sidebar" style="position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); border: 0;"></nav>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<main aria-label="Analysis Main Content" id="main-content-anchor" tabindex="-1" style="position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); border: 0;"></main>',
        unsafe_allow_html=True
    )

    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Accessibility")
    if "accessibility_mode" not in st.session_state:
        st.session_state["accessibility_mode"] = False

    toggle_fn = getattr(st.sidebar, "toggle", lambda *args, **kwargs: False)
    toggle_fn(
        "Accessibility Mode",
        key="accessibility_mode",
        help="Enable high-fidelity text alternatives for screen readers.",
    )

    st.sidebar.header("Select Trial Design")
    available_designs = PROTOCOL_REGISTRY.get_designs()
    if not available_designs:
        st.error("No trial designs registered.")
        return

    design_type = create_widget(
        st,
        "selectbox",
        "design_type",
        "Choose the type of trial design for your simulation results:",
        tuple(available_designs),
    )

    from clintrials.visualization.dashboard.factory import REGISTRY

    with st.sidebar.expander("View Glossary"):
        # Aggregate relevant fields for the selected view
        entries = []
        entries.append(
            (
                "Choose the type of trial design for your simulation results:",
                REGISTRY.get("design_type", ""),
            )
        )

        preview_func = PROTOCOL_REGISTRY.get_preview(design_type)
        if preview_func is None:
            # Typically Win Ratio
            try:
                from clintrials.core.schema import WinRatioSchema
                for name, field in WinRatioSchema.model_fields.items():
                    if name in REGISTRY:
                        entries.append((field.description, REGISTRY[name]))
            except ImportError:
                pass
            entries.append(
                ("Run Simulation", REGISTRY.get("run_simulation_button", ""))
            )
        else:
            entries.append(
                (
                    "Upload a JSON file with simulation results",
                    REGISTRY.get("uploaded_file", ""),
                )
            )
            if "true_tox" in REGISTRY:
                entries.append(("true_tox", REGISTRY["true_tox"]))
            if "true_prob_tox" in REGISTRY:
                entries.append(("true_prob_tox", REGISTRY["true_prob_tox"]))
            if "true_prob_eff" in REGISTRY:
                entries.append(("true_prob_eff", REGISTRY["true_prob_eff"]))

        for label, desc in entries:
            st.markdown(f"**{label}**: {desc}")

    render_func = PROTOCOL_REGISTRY.get_render(design_type)

    if PROTOCOL_REGISTRY.get_preview(design_type) is None:
        if render_func:
            render_func()
    else:
        st.sidebar.header("Data Mode")
        data_mode = st.sidebar.radio(
            "Select Data Source", 
            ["Preview Mode", "Manual JSON Upload"], 
            help="Switch between automatically generated preview simulations and manual file upload."
        )

        if data_mode == "Manual JSON Upload":
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
                if render_func:
                    render_func(sims)
        else:
            st.sidebar.header("Preview Parameters")
            target_tox = st.sidebar.number_input("Target Toxicity", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
            cohort_size = st.sidebar.number_input("Cohort Size", min_value=1, max_value=10, value=3)
            max_size = st.sidebar.number_input("Sample Size (N)", min_value=10, max_value=100, value=60, step=10)
            
            try:
                sims = get_preview_sims(design_type, target_tox, cohort_size, max_size)
                if render_func:
                    render_func(sims)
            except Exception as e:
                st.error(f"Simulation failed with the selected parameters: {e}")


if __name__ == "__main__":
    main()


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY

    __doc__ = __doc__.format(**REGISTRY)
