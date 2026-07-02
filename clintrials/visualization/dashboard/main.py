"""
Main entry point for the Streamlit dashboard.


Random Seed Strategy: {main_seed_strategy}
"""

import json
import logging

import streamlit as st

from clintrials.visualization.dashboard.factory import create_widget
from clintrials.visualization.dashboard.views import (
    crm_view,
    efftox_view,
    watu_view,
    winratio_view,
)

logger = logging.getLogger(__name__)

def setup_persistence():
    """Sets up local storage sync and indexedDB auto-loading."""
    has_js = False
    try:
        import js
        from pyodide.ffi import create_proxy
        has_js = True
    except ImportError:
        pass

    if not has_js:
        return

    # Check URL params for explicit overrides
    url_params = {}
    if hasattr(st, "query_params"):
        url_params = st.query_params
    else:
        try:
            url_params = st.experimental_get_query_params()
        except Exception:
            pass

    # UI State Persistence (localStorage)
    if "accessibility_mode" not in st.session_state:
        stored_acc = js.window.localStorage.getItem("accessibility_mode")
        url_acc = url_params.get("accessibility_mode", [None])[0] if isinstance(url_params.get("accessibility_mode"), list) else url_params.get("accessibility_mode")
        
        if url_acc is not None:
            st.session_state["accessibility_mode"] = url_acc.lower() == "true"
        elif stored_acc is not None:
            st.session_state["accessibility_mode"] = stored_acc == "true"
        else:
            st.session_state["accessibility_mode"] = False
            
    if "design_type" not in st.session_state:
        stored_design = js.window.localStorage.getItem("design_type")
        url_design = url_params.get("design_type", [None])[0] if isinstance(url_params.get("design_type"), list) else url_params.get("design_type")
        
        if url_design is not None:
            st.session_state["design_type"] = url_design
        elif stored_design is not None:
            st.session_state["design_type"] = stored_design
            
    # Auto-load simulation data from IndexedDB
    if "idb_loaded" not in st.session_state:
        st.session_state["idb_loaded"] = False
        st.session_state["idb_data"] = []
        
        def on_idb_load(data_str):
            try:
                results = json.loads(data_str)
                combined_sims = []
                for record in results:
                    batch = record.get("batch", [])
                    if isinstance(batch, list):
                        combined_sims.extend(batch)
                    else:
                        sims = batch.get("Simulations", [])
                        combined_sims.extend(sims)
                
                st.session_state["idb_data"] = combined_sims
            except Exception as e:
                logger.error(f"Error parsing IDB data: {e}")
            finally:
                st.session_state["idb_loaded"] = True
                if st.session_state.get("idb_data", []):
                    # Trigger a rerun automatically by clicking the refresh button
                    try:
                        import js
                        js.eval("""
                        window.setTimeout(() => {
                            try {
                                // In Stlite, components might be in the same document
                                let doc = window.document;
                                let buttons = doc.querySelectorAll('button');
                                for (let b of buttons) {
                                    if (b.innerText && b.innerText.includes('Refresh Persistent Data')) {
                                        b.click();
                                        return;
                                    }
                                }
                                // Fallback for iframe environment
                                if (window.parent && window.parent !== window) {
                                    let parentButtons = window.parent.document.querySelectorAll('button');
                                    for (let b of parentButtons) {
                                        if (b.innerText && b.innerText.includes('Refresh Persistent Data')) {
                                            b.click();
                                            return;
                                        }
                                    }
                                }
                            } catch (e) {}
                        }, 500);
                        """)
                    except Exception:
                        pass
                
        try:
            js.window._on_idb_load = create_proxy(on_idb_load)
            js.eval("""
            (function() {
                try {
                    var req = window.indexedDB.open('clintrials_db', 1);
                    req.onsuccess = function(e) {
                        var db = e.target.result;
                        if (!db.objectStoreNames.contains('simulations')) {
                            window._on_idb_load('[]');
                            return;
                        }
                        var tx = db.transaction('simulations', 'readonly');
                        var store = tx.objectStore('simulations');
                        var allReq = store.getAll();
                        allReq.onsuccess = function() {
                            window._on_idb_load(JSON.stringify(allReq.result));
                        };
                        allReq.onerror = function() { window._on_idb_load('[]'); };
                    };
                    req.onerror = function() { window._on_idb_load('[]'); };
                } catch(err) {
                    window._on_idb_load('[]');
                }
            })();
            """)
        except Exception:
            st.session_state["idb_loaded"] = True

def sync_state_to_storage():
    """Syncs the current Streamlit UI state to localStorage."""
    try:
        import js
        acc_mode = str(st.session_state.get("accessibility_mode", False)).lower()
        design = st.session_state.get("design_type", "CRM")
        
        js.eval(f"""
        try {{
            window.localStorage.setItem('accessibility_mode', '{acc_mode}');
            window.localStorage.setItem('design_type', '{design}');
        }} catch (e) {{
            if (e.name === 'QuotaExceededError') {{
                console.warn('Storage quota exceeded');
            }}
        }}
        """)
    except ImportError:
        pass


def main():
    """Sets up the Streamlit dashboard and renders the appropriate view."""
    if not hasattr(st, "session_state"):
        st.session_state = {}
        
    setup_persistence()
    
    st.title("Interactive Simulation Dashboard")

    st.sidebar.header("Accessibility")

    toggle_fn = getattr(st.sidebar, "toggle", lambda *args, **kwargs: False)
    acc_mode = toggle_fn(
        "Accessibility Mode",
        value=st.session_state.get("accessibility_mode", False),
        key="accessibility_mode",
        help="Enable high-fidelity text alternatives for screen readers.",
    )
    
    st.sidebar.header("Select Trial Design")
    
    # Get current index for selectbox
    design_options = ("CRM", "EffTox", "WATU", "Win Ratio")
    current_design = st.session_state.get("design_type", "CRM")
    if current_design not in design_options:
        current_design = "CRM"
    
    design_type = create_widget(
        st,
        "selectbox",
        "design_type",
        "Choose the type of trial design for your simulation results:",
        design_options,
        index=design_options.index(current_design),
        key="design_type"
    )

    sync_state_to_storage()

    from clintrials.visualization.dashboard.factory import REGISTRY

    with st.sidebar.expander("View Glossary"):
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

    # Clear Session Feature
    if st.sidebar.button("Clear Session", type="primary"):
        try:
            import js
            js.eval("""
            try {
                window.localStorage.clear();
                window.indexedDB.deleteDatabase('clintrials_db');
            } catch(e) { console.error(e); }
            """)
        except ImportError:
            pass
        st.session_state.clear()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    # Load Persistent Data button (if data was loaded async and we need user interaction to refresh)
    if st.session_state.get("idb_loaded", False) and st.session_state.get("idb_data", []):
        st.sidebar.success(f"Recovered {len(st.session_state['idb_data'])} persistent simulations.")

    if not st.session_state.get("idb_loaded", False):
        # We provide a manual refresh button if async load takes time
        if st.sidebar.button("Refresh Persistent Data"):
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    if design_type == "Win Ratio":
        winratio_view.render()
    else:
        sims = None
        
        # Priority to recovered session data
        if st.session_state.get("idb_loaded", False) and st.session_state.get("idb_data", []):
            sims = st.session_state["idb_data"]
            st.info("Loaded simulation results from persistent browser storage.")
        
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
            st.sidebar.success(f"Successfully loaded {len(sims)} simulations from file.")

        if sims is not None:
            if design_type == "CRM":
                crm_view.render(sims)
            elif design_type == "EffTox":
                efftox_view.render(sims)
            elif design_type == "WATU":
                watu_view.render(sims)

if __name__ == "__main__":
    main()


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY

    __doc__ = __doc__.format(**REGISTRY)
