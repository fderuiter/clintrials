"""Main entry point for the Streamlit dashboard.

Random Seed Strategy: {main_seed_strategy}
"""

import json

import streamlit as st

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.visualization.dashboard.factory import create_widget


@st.cache_data(show_spinner=False)
def get_preview_sims(design_type, target_tox, cohort_size, max_size):
    """Run and cache default preview simulations based on the selected design type and parameters."""
    preview_func = PROTOCOL_REGISTRY.get_preview(design_type)
    if preview_func:
        with st.spinner(f"Running Default Preview Simulation for {design_type}..."):
            return preview_func(target_tox, cohort_size, max_size)
    return []

def main():
    """Sets up the Streamlit dashboard and renders the appropriate view."""
    try:
        import streamlit.components.v1 as components
        components.html("""
        <script>
            let parentDoc;
            try {
                parentDoc = window.parent.document;
                // Force access to trigger cross-origin exception if any
                const test = parentDoc.domain;
            } catch (e) {
                console.warn('Cross-origin or sandbox iframe blocked parent document access:', e);
            }

            if (parentDoc) {
                try {
                    // Skip-link Injection
                    if (!parentDoc.getElementById('skip-link-style')) {
                        const style = parentDoc.createElement('style');
                        style.id = 'skip-link-style';
                        style.innerHTML = `
                        .skip-link {
                          position: absolute;
                          width: 1px;
                          height: 1px;
                          padding: 0;
                          margin: -1px;
                          overflow: hidden;
                          clip: rect(0, 0, 0, 0);
                          white-space: nowrap;
                          border: 0;
                        }
                        .skip-link:focus {
                          width: auto;
                          height: auto;
                          padding: 10px 15px;
                          margin: 0;
                          overflow: visible;
                          clip: auto;
                          white-space: normal;
                          background-color: #496D89;
                          color: #FFFFFF;
                          z-index: 999999;
                          top: 0;
                          left: 0;
                          text-decoration: none;
                          font-family: sans-serif;
                        }
                        `;
                        parentDoc.head.appendChild(style);
                    }

                    if (!parentDoc.getElementById('skip-link-anchor')) {
                        const skipLink = parentDoc.createElement('a');
                        skipLink.id = 'skip-link-anchor';
                        skipLink.className = 'skip-link';
                        skipLink.href = '#main-content-anchor';
                        skipLink.innerText = 'Skip to main content';
                        parentDoc.body.insertBefore(skipLink, parentDoc.body.firstChild);
                    }

                    // Fix aria-allowed-attr for sidebar
                    const sidebars = parentDoc.querySelectorAll('.stSidebar');
                    sidebars.forEach(sidebar => {
                        sidebar.removeAttribute('aria-expanded');
                        sidebar.setAttribute('role', 'navigation');
                    });

                    // Fix region issues by wrapping main content or adding roles
                    const mainContainer = parentDoc.querySelector('.stMain');
                    if (mainContainer) {
                        mainContainer.setAttribute('role', 'main');
                        mainContainer.id = 'main-content-anchor';
                        mainContainer.setAttribute('tabindex', '-1');
                    }

                    const appHeader = parentDoc.querySelector('header[data-testid="stHeader"]');
                    if (appHeader) {
                        appHeader.setAttribute('role', 'banner');
                    }

                    // Fix scrollable-region-focusable for JSON and code blocks
                    const scrollables = parentDoc.querySelectorAll('.stJson, .stCodeBlock');
                    scrollables.forEach(el => {
                        el.setAttribute('tabindex', '0');
                    });
                } catch (e) {
                    console.error('Error applying same-origin accessibility adjustments:', e);
                }

                // Periodically run to catch dynamically rendered elements
                setInterval(() => {
                    try {
                        const pDoc = window.parent.document;
                        pDoc.querySelectorAll('.stSidebar').forEach(s => s.removeAttribute('aria-expanded'));
                        pDoc.querySelectorAll('.stJson').forEach(j => {
                            if (!j.hasAttribute('tabindex')) j.setAttribute('tabindex', '0');
                        });

                        const mainContainer = pDoc.querySelector('.stMain');
                        if (mainContainer) {
                            mainContainer.setAttribute('role', 'main');
                            mainContainer.id = 'main-content-anchor';
                            mainContainer.setAttribute('tabindex', '-1');
                        }

                        if (!pDoc.getElementById('skip-link-anchor')) {
                            const skipLink = pDoc.createElement('a');
                            skipLink.id = 'skip-link-anchor';
                            skipLink.className = 'skip-link';
                            skipLink.href = '#main-content-anchor';
                            skipLink.innerText = 'Skip to main content';
                            pDoc.body.insertBefore(skipLink, pDoc.body.firstChild);
                        }
                    } catch (e) {
                        console.warn('Cross-origin or sandbox iframe blocked parent document access during interval:', e);
                    }
                }, 1000);
            }
        </script>
        """, height=0, width=0)
    except ImportError:
        pass

    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Accessibility Settings")
    st.session_state["accessibility_mode"] = st.sidebar.checkbox(
        "Enable Screen-Reader Optimized Mode",
        value=st.session_state.get("accessibility_mode", False),
        help="Restructures large tables into hierarchical nested details for easier navigation."
    )

    st.sidebar.header("Select Trial Design")
    available_designs = PROTOCOL_REGISTRY.get_designs()
    if not available_designs:
        st.error("No trial designs registered.")
        return

    config = {}
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except Exception:
        pass

    active_view = config.get("view", "")
    default_index = 0
    if active_view and active_view in available_designs:
        default_index = list(available_designs).index(active_view)

    is_embed = str(config.get("embed", "")).lower() == "true"
    if is_embed:
        st.markdown(
            """
            <style>
                header[data-testid="stHeader"] { display: none !important; }
                footer { display: none !important; }
                section[data-testid="stSidebar"] { display: none !important; }
                .stDeployButton { display: none !important; }
                div[data-testid="collapsedControl"] { display: none !important; }
                .stApp { background-color: transparent !important; }
            </style>
            """,
            unsafe_allow_html=True
        )


    design_type = create_widget(
        st,
        "selectbox",
        "design_type",
        "Choose the type of trial design for your simulation results:",
        tuple(available_designs),
        index=default_index,
    )

    from clintrials.visualization.dashboard.factory import UI_REGISTRY

    with st.sidebar.expander("View Glossary"):
        # Aggregate relevant fields for the selected view
        entries = []
        entries.append(
            (
                "Choose the type of trial design for your simulation results:",
                UI_REGISTRY.get("design_type", ""),
            )
        )

        preview_func = PROTOCOL_REGISTRY.get_preview(design_type)
        if preview_func is None:
            # Typically Win Ratio
            try:
                from clintrials.core.schema import WinRatioSchema
                for name, field in WinRatioSchema.model_fields.items():
                    if name in UI_REGISTRY:
                        entries.append((field.description, UI_REGISTRY[name]))
            except ImportError:
                pass
            entries.append(
                ("Run Simulation", UI_REGISTRY.get("run_simulation_button", ""))
            )
        else:
            entries.append(
                (
                    "Upload a JSON file with simulation results",
                    UI_REGISTRY.get("uploaded_file", ""),
                )
            )
            if "true_tox" in UI_REGISTRY:
                entries.append(("true_tox", UI_REGISTRY["true_tox"]))
            if "true_prob_tox" in UI_REGISTRY:
                entries.append(("true_prob_tox", UI_REGISTRY["true_prob_tox"]))
            if "true_prob_eff" in UI_REGISTRY:
                entries.append(("true_prob_eff", UI_REGISTRY["true_prob_eff"]))

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

            from clintrials.visualization.dashboard.utils import announce_status_locally
            try:
                announce_status_locally("Simulation in progress", key="preview-start")
                sims = get_preview_sims(design_type, target_tox, cohort_size, max_size)
                announce_status_locally("Simulation completed", key="preview-complete")
                if render_func:
                    render_func(sims)
            except Exception as e:
                announce_status_locally("Simulation failed", key="preview-fail")
                st.error(f"Simulation failed with the selected parameters: {e}")


if __name__ == "__main__":
    main()


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY

    __doc__ = __doc__.format(**CORE_REGISTRY)
