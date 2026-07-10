"""
Main entry point for the Streamlit dashboard.


Random Seed Strategy: {main_seed_strategy}
"""

import json

import streamlit as st

from clintrials.visualization.dashboard.factory import create_widget
from clintrials.visualization.dashboard.views import (
    crm_view,
    efftox_view,
    watu_view,
    winratio_view,
)


@st.cache_data(show_spinner=False)
def get_preview_sims(design_type, target_tox, cohort_size, max_size):
    """
    Run and cache default preview simulations based on the selected design type and parameters.
    
    Args:
        design_type: The type of clinical trial design (e.g., 'CRM').
        target_tox: The target toxicity level.
        cohort_size: The number of patients per cohort.
        max_size: The maximum total sample size for the trial.
        
    Returns:
        list: A list of simulation report dictionaries.
    """
    with st.spinner(f"Running Default Preview Simulation for {design_type}..."):
        sims = []
        if design_type == "CRM":
            from clintrials.dosefinding.crm import CRM
            from clintrials.dosefinding import simulate_dose_finding_trial
            
            crm = CRM(
                prior=[0.05, 0.1, 0.2, 0.3, 0.4], 
                target=target_tox, 
                first_dose=1, 
                max_size=max_size
            )
            scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4, 0.5)]
            for true_tox in scenarios:
                for _ in range(20):
                    report = simulate_dose_finding_trial(crm, true_toxicities=true_tox, cohort_size=cohort_size)
                    report["true_tox"] = true_tox
                    sims.append(report)
                    
        elif design_type == "EffTox":
            from clintrials.dosefinding.efftox import EffTox, LpNormCurve
            from clintrials.dosefinding.efficacytoxicity import simulate_trial
            
            real_doses = [1.0, 2.0, 3.0, 4.0, 5.0]
            prior_tox_probs = [0.05, 0.1, 0.2, 0.3, 0.4]
            prior_eff_probs = [0.2, 0.4, 0.6, 0.7, 0.8]
            
            metric = LpNormCurve(0.2, 0.4, 0.5, 0.2)
            trial = EffTox(
                real_doses=real_doses,
                prior_tox_probs=prior_tox_probs,
                prior_eff_probs=prior_eff_probs,
                tox_cutoff=0.4,
                eff_cutoff=0.2,
                tox_certainty=0.8,
                eff_certainty=0.8,
                metric=metric,
                max_size=max_size,
            )
            
            tox_scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4)]
            eff_scenarios = [(0.2, 0.3, 0.4, 0.5, 0.6)]
            
            for t_tox in tox_scenarios:
                for t_eff in eff_scenarios:
                    for _ in range(10):
                        report = simulate_trial(trial, true_toxicities=t_tox, true_efficacies=t_eff, cohort_size=cohort_size)
                        report["true_prob_tox"] = t_tox
                        report["true_prob_eff"] = t_eff
                        sims.append(report)

        elif design_type == "WATU":
            from clintrials.dosefinding.watu import WATU
            from clintrials.dosefinding.efftox import LpNormCurve
            from clintrials.dosefinding.efficacytoxicity import simulate_trial
            skeletons = [
                [0.60, 0.50, 0.40, 0.30, 0.20],
                [0.50, 0.60, 0.50, 0.40, 0.30],
                [0.40, 0.50, 0.60, 0.50, 0.40],
                [0.30, 0.40, 0.50, 0.60, 0.50],
                [0.20, 0.30, 0.40, 0.50, 0.60],
            ]
            tox_prior = [0.05, 0.1, 0.2, 0.3, 0.4]
            metric = LpNormCurve(0.2, 0.4, 0.5, 0.2)
            
            watu = WATU(
                skeletons=skeletons,
                prior_tox_probs=tox_prior,
                tox_target=target_tox,
                tox_limit=0.4,
                eff_limit=0.2,
                metric=metric,
                first_dose=1,
                max_size=max_size
            )
            
            tox_scenarios = [(0.05, 0.1, 0.2, 0.3, 0.4)]
            eff_scenarios = [(0.2, 0.3, 0.4, 0.5, 0.6)]
            for t_tox in tox_scenarios:
                for t_eff in eff_scenarios:
                    for _ in range(10):
                        report = simulate_trial(watu, true_toxicities=t_tox, true_efficacies=t_eff, cohort_size=cohort_size)
                        report["true_prob_tox"] = t_tox
                        report["true_prob_eff"] = t_eff
                        sims.append(report)
                        
        return sims


def main():
    """Sets up the Streamlit dashboard and renders the appropriate view.

    This function creates the main layout of the dashboard, including the
    sidebar for selecting the trial design and uploading simulation results.
    It then calls the appropriate render function based on the user's
    selection.
    """
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
    if not hasattr(st, "session_state"):
        st.session_state = {}
    if "accessibility_mode" not in st.session_state:
        st.session_state["accessibility_mode"] = False

    st.sidebar.toggle(
        "Accessibility Mode",
        key="accessibility_mode",
        help="Enable high-fidelity text alternatives for screen readers.",
    )

    st.sidebar.header("Select Trial Design")
    design_type = create_widget(
        st,
        "selectbox",
        "design_type",
        "Choose the type of trial design for your simulation results:",
        ("CRM", "EffTox", "WATU", "Win Ratio"),
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

        if design_type == "Win Ratio":
            from clintrials.core.schema import WinRatioSchema

            for name, field in WinRatioSchema.model_fields.items():
                if name in REGISTRY:
                    entries.append((field.description, REGISTRY[name]))
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
            if design_type == "CRM":
                if "true_tox" in REGISTRY:
                    entries.append(("true_tox", REGISTRY["true_tox"]))
            elif design_type == "EffTox" or design_type == "WATU":
                if "true_prob_tox" in REGISTRY:
                    entries.append(("true_prob_tox", REGISTRY["true_prob_tox"]))
                if "true_prob_eff" in REGISTRY:
                    entries.append(("true_prob_eff", REGISTRY["true_prob_eff"]))

        for label, desc in entries:
            st.markdown(f"**{label}**: {desc}")

    if design_type == "Win Ratio":
        winratio_view.render()
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

                if design_type == "CRM":
                    crm_view.render(sims)
                elif design_type == "EffTox":
                    efftox_view.render(sims)
                elif design_type == "WATU":
                    watu_view.render(sims)
        else:
            st.sidebar.header("Preview Parameters")
            target_tox = st.sidebar.number_input("Target Toxicity", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
            cohort_size = st.sidebar.number_input("Cohort Size", min_value=1, max_value=10, value=3)
            max_size = st.sidebar.number_input("Sample Size (N)", min_value=10, max_value=100, value=60, step=10)
            
            try:
                sims = get_preview_sims(design_type, target_tox, cohort_size, max_size)
                if design_type == "CRM":
                    crm_view.render(sims)
                elif design_type == "EffTox":
                    efftox_view.render(sims)
                elif design_type == "WATU":
                    watu_view.render(sims)
            except Exception as e:
                st.error(f"Simulation failed with the selected parameters: {e}")


if __name__ == "__main__":
    main()


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY

    __doc__ = __doc__.format(**REGISTRY)
