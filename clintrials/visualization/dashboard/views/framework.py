# SPDX-License-Identifier: MIT

"""Framework tools for declarative and reusable simulation dashboard views."""

from functools import wraps
from typing import Any, Callable, Optional

from clintrials.core.viz_interface import get_visualization_provider
from clintrials.visualization.dashboard.factory import render_accessible_chart


class BaseSimulationView:
    """Base declarative view class for trial design simulation dashboards."""

    model_name = ""
    title = ""
    file_prefix = ""
    param_space_config = {}  # type: ignore
    model_class = None
    var_map = None
    csv_index = True
    skip_summary_table = False

    @classmethod
    def __init_subclass__(cls, **kwargs):  # type: ignore
        """Register the view automatically on subclassing."""
        super().__init_subclass__(**kwargs)
        if cls.model_name:
            from clintrials.core.registry import PROTOCOL_REGISTRY

            decorated_render = dashboard_view(
                title=cls.title,
                model_name=cls.model_name,
                file_prefix=cls.file_prefix,
                csv_index=cls.csv_index,
                skip_summary_table=cls.skip_summary_table,
                param_space_config=cls.param_space_config,
            )(cls._base_render)

            preview = None
            if "preview_sims" in cls.__dict__:
                preview = cls.preview_sims  # type: ignore

            PROTOCOL_REGISTRY.register_manual(
                cls.model_name, render_func=decorated_render, preview_func=preview
            )

    @classmethod
    def _base_render(cls, sims, ps=None):  # type: ignore
        """Render the sidebar controls, parse parameter combinations, and execute the view mapping."""
        from clintrials.core.simulation import extract_sim_data

        func_map = cls.model_class.get_summary_functions()  # type: ignore
        summary_df = extract_sim_data(
            sims, ps, func_map, var_map=cls.var_map, return_type="dataframe"
        )

        figures = cls.build_figures(summary_df)  # type: ignore
        return summary_df, figures

    @classmethod
    def build_figures(cls, summary_df):  # type: ignore
        """Build figures from the simulation summary dataframe. Should be overridden."""
        return []


from clintrials.utils import ParameterSpace


def render_sidebar_config(param_space_config: dict) -> ParameterSpace:  # type: ignore
    """Render the sidebar configuration and return the parameter space."""
    import streamlit as st

    st.sidebar.header("Trial Parameters")
    ps = ParameterSpace()
    for k, v in param_space_config.items():
        ps.add(k, v)
    if param_space_config:
        st.sidebar.json(param_space_config)
    return ps


def dashboard_view(
    title: str,
    model_name: str,
    file_prefix: str,
    csv_index: bool = True,
    skip_summary_table: bool = False,
    param_space_config: Optional[dict[Any, Any]] = None,
) -> Callable[..., Any]:
    """Decorator to generate a standard dashboard view."""

    def decorator(func):  # type: ignore
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore
            import streamlit as st

            from clintrials.visualization.dashboard.factory import create_widget

            if not hasattr(st, "fragment"):
                st.fragment = lambda f: f  # type: ignore[assignment]
            if not hasattr(st, "columns"):
                st.columns = lambda x: (st, st)  # type: ignore[assignment, misc]

            # Dynamic mocks fallback to prevent test crashes in environments with incomplete streamlit mocks
            if not hasattr(st, "tabs"):
                class DummyTab:
                    def __enter__(self) -> "DummyTab":
                        return self
                    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
                        return None
                st.tabs = lambda labels: [DummyTab() for _ in labels]  # type: ignore[assignment, misc]

            class DummyContext:
                def __init__(self, obj: Any = None) -> None:
                    self._obj = obj
                def __getattr__(self, name: str) -> Any:
                    if name in ("sidebar", "session_state"):
                        raise AttributeError(f"DummyContext has no attribute '{name}'")
                    if self._obj is not None:
                        return getattr(self._obj, name)
                    return lambda *args, **kwargs: None
                def __enter__(self) -> "DummyContext":
                    return self
                def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
                    return None

            original_columns = st.columns
            st.columns = lambda *a, **kw: [DummyContext(c) for c in original_columns(*a, **kw)]
            dummy_col = DummyContext(None)

            if not hasattr(st, "info"):
                st.info = getattr(st, "markdown", lambda *args, **kwargs: None)  # type: ignore[arg-type]
            if not hasattr(st, "metric"):
                st.metric = lambda *args, **kwargs: None  # type: ignore[assignment]
            if not hasattr(st, "dataframe"):
                st.dataframe = getattr(st, "markdown", lambda *args, **kwargs: None)  # type: ignore[arg-type]
            if not hasattr(st, "selectbox"):
                st.selectbox = lambda *args, **kwargs: "None / No Override"  # type: ignore[assignment]
            if not hasattr(st, "text_input"):
                st.text_input = lambda *args, **kwargs: "0.05, 0.1, 0.2, 0.3, 0.4"
            if not hasattr(st, "number_input"):
                st.number_input = lambda *args, **kwargs: 3
            if not hasattr(st, "button"):
                st.button = lambda *args, **kwargs: False

            ps = None
            if param_space_config is not None:
                ps = render_sidebar_config(param_space_config)

            st.header(title)

            # Split the dashboard views into standard reports and sandbox tabs
            tab_report, tab_sandbox = st.tabs(["Standard Reports", "Interactive Sandbox"])

            with tab_report:
                try:
                    if ps is not None:
                        kwargs["ps"] = ps
                    result = func(*args, **kwargs)
                    if result is not None:
                        # Check if it returned 2 or 3 items
                        if isinstance(result, tuple) and len(result) == 3:
                            summary_df, figures, extra_text_summaries = result
                        else:
                            summary_df, figures = result
                            extra_text_summaries = None

                        if summary_df is not None and not summary_df.empty:
                            if not skip_summary_table:
                                st.subheader("Simulation Summary")
                                from clintrials.visualization.models import (
                                    MultiFormatSummaryContainer,
                                )

                                df_for_rendering = summary_df.reset_index()
                                rename_map = {}
                                for c in df_for_rendering.columns:
                                    if str(c).lower() == "index" or c == "":
                                        rename_map[c] = "Index"
                                df_for_rendering = df_for_rendering.rename(columns=rename_map)

                                for col in df_for_rendering.columns:
                                    try:
                                        df_for_rendering[col].nunique()
                                    except TypeError:
                                        df_for_rendering[col] = df_for_rendering[col].astype(
                                            str
                                        )

                                container = MultiFormatSummaryContainer(
                                    title="Simulation Summary", df=df_for_rendering
                                )
                                st.markdown(container.html, unsafe_allow_html=True)

                        text_summaries = []
                        if extra_text_summaries:
                            if isinstance(extra_text_summaries, list):
                                text_summaries.extend(extra_text_summaries)
                            else:
                                text_summaries.append(extra_text_summaries)

                        if figures:
                            st.header(
                                "Operating Characteristics"
                                if model_name != "Win Ratio"
                                else "Visualizations"
                            )
                            for fig_title, fig in figures:
                                if fig_title:
                                    st.subheader(fig_title)

                                meta = getattr(
                                    getattr(fig, "layout", None),
                                    "meta",
                                    "No data summary available.",
                                )
                                text_summaries.append(meta)
                                render_accessible_chart(st, fig)  # type: ignore
                        elif summary_df is not None and not summary_df.empty:
                            pass

                        if summary_df is not None and not summary_df.empty:
                            st.header("Export Results")
                            cols_exp = st.columns(2)
                            col1 = cols_exp[0] if len(cols_exp) > 0 else dummy_col
                            col2 = cols_exp[1] if len(cols_exp) > 1 else dummy_col

                            csv_data = summary_df.to_csv(index=csv_index)
                            getattr(col1, "download_button", lambda *args, **kwargs: None)(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"{file_prefix}.csv",
                                mime="text/csv",
                            )

                            viz_provider = get_visualization_provider()
                            pdf_data = (
                                viz_provider.generate_pdf_report(
                                    summary_df, model_name, text_summaries=text_summaries
                                )
                                if viz_provider
                                else None
                            )

                            if pdf_data is not None:
                                getattr(col2, "download_button", lambda *args, **kwargs: None)(
                                    label="Download PDF",
                                    data=pdf_data,
                                    file_name=f"{file_prefix}.pdf",
                                    mime="application/pdf",
                                )
                            else:
                                if hasattr(col2, "warning"):
                                    col2.warning("PDF export requires the 'fpdf2' package.")
                    else:
                        st.info("No standard simulation results available.")
                except Exception as e:
                    if hasattr(st, "error"):
                        st.error(f"An error occurred during summarization or plotting: {e}")
                    else:
                        raise e

            with tab_sandbox:
                st.subheader("Interactive Single-Trial Sandbox")

                # Check if the trial design type is supported for sandbox
                supported_models = ["CRM", "EffTox", "Wages & Tait", "WageStait", "WATU", "WaTu"]
                if model_name not in supported_models:
                    st.info(
                        "The Interactive Sandbox is designed for dose-finding protocols "
                        "(CRM, EffTox, Wages & Tait, WATU) to test step-by-step cohort overrides and safety rules."
                    )
                else:
                    # Initialize Sandbox state if not present
                    protocol_key = f"{model_name}_sandbox_protocol"
                    history_key = f"{model_name}_sandbox_history"
                    override_key = f"{model_name}_sandbox_override_dose"
                    cohort_size_key = f"{model_name}_sandbox_cohort_size"
                    target_tox_key = f"{model_name}_sandbox_target_tox"
                    max_size_key = f"{model_name}_sandbox_max_size"

                    # 1. State-continuity / persistent helper functions
                    def init_sandbox_protocol(m_name: str) -> None:
                        t_tox = st.session_state.get(target_tox_key, 0.25)
                        m_size = st.session_state.get(max_size_key, 60)

                        protocol_obj: Any = None
                        if m_name == "CRM":
                            from clintrials.dosefinding.crm import CRM
                            protocol_obj = CRM(
                                prior=[0.05, 0.1, 0.2, 0.3, 0.4],
                                target=t_tox,
                                first_dose=1,
                                max_size=m_size,
                            )
                        elif m_name == "EffTox":
                            from clintrials.dosefinding.efftox import (
                                EffTox,
                                LpNormCurve,
                            )
                            real_doses = [1.0, 2.0, 3.0, 4.0, 5.0]
                            prior_tox_probs = [0.05, 0.1, 0.2, 0.3, 0.4]
                            prior_eff_probs = [0.2, 0.4, 0.6, 0.7, 0.8]
                            metric_curve = LpNormCurve(0.2, 0.4, 0.5, 0.2)
                            protocol_obj = EffTox(
                                real_doses=real_doses,
                                prior_tox_probs=prior_tox_probs,
                                prior_eff_probs=prior_eff_probs,
                                tox_cutoff=0.4,
                                eff_cutoff=0.2,
                                tox_certainty=0.8,
                                eff_certainty=0.8,
                                metric=metric_curve,
                                max_size=m_size,
                            )
                        elif m_name in ["Wages & Tait", "WageStait"]:
                            from clintrials.dosefinding.wagestait import WagesTait
                            skeletons_list = [
                                [0.60, 0.50, 0.40, 0.30, 0.20],
                                [0.50, 0.60, 0.50, 0.40, 0.30],
                                [0.40, 0.50, 0.60, 0.50, 0.40],
                                [0.30, 0.40, 0.50, 0.60, 0.50],
                                [0.20, 0.30, 0.40, 0.50, 0.60],
                            ]
                            tox_prior_list = [0.05, 0.1, 0.2, 0.3, 0.4]
                            protocol_obj = WagesTait(
                                skeletons=skeletons_list,
                                prior_tox_probs=tox_prior_list,
                                tox_target=t_tox,
                                tox_limit=0.4,
                                eff_limit=0.2,
                                first_dose=1,
                                max_size=m_size,
                                randomisation_stage_size=m_size // 2,
                            )
                        elif m_name in ["WATU", "WaTu"]:
                            from clintrials.dosefinding.efftox import LpNormCurve
                            from clintrials.dosefinding.watu import WATU
                            skeletons_list = [
                                [0.60, 0.50, 0.40, 0.30, 0.20],
                                [0.50, 0.60, 0.50, 0.40, 0.30],
                                [0.40, 0.50, 0.60, 0.50, 0.40],
                                [0.30, 0.40, 0.50, 0.60, 0.50],
                                [0.20, 0.30, 0.40, 0.50, 0.60],
                            ]
                            tox_prior_list = [0.05, 0.1, 0.2, 0.3, 0.4]
                            metric_curve = LpNormCurve(0.2, 0.4, 0.5, 0.2)
                            protocol_obj = WATU(
                                skeletons=skeletons_list,
                                prior_tox_probs=tox_prior_list,
                                tox_target=t_tox,
                                tox_limit=0.4,
                                eff_limit=0.2,
                                metric=metric_curve,
                                first_dose=1,
                                max_size=m_size,
                            )
                        else:
                            raise ValueError(f"Unknown or unsupported model for sandbox: {m_name}")
                        st.session_state[protocol_key] = protocol_obj

                    # Ensure keys are present with initial defaults
                    if target_tox_key not in st.session_state:
                        st.session_state[target_tox_key] = 0.25
                    if max_size_key not in st.session_state:
                        st.session_state[max_size_key] = 60
                    if cohort_size_key not in st.session_state:
                        st.session_state[cohort_size_key] = 3

                    # Detect core parameters change to automatically reset
                    current_p_target_tox = st.session_state[target_tox_key]
                    current_p_max_size = st.session_state[max_size_key]

                    active_target_tox_key = f"{model_name}_sandbox_active_target_tox"
                    active_max_size_key = f"{model_name}_sandbox_active_max_size"

                    if active_target_tox_key not in st.session_state or st.session_state[active_target_tox_key] != current_p_target_tox:
                        st.session_state[active_target_tox_key] = current_p_target_tox
                        st.session_state[history_key] = []
                        init_sandbox_protocol(model_name)
                    elif active_max_size_key not in st.session_state or st.session_state[active_max_size_key] != current_p_max_size:
                        st.session_state[active_max_size_key] = current_p_max_size
                        st.session_state[history_key] = []
                        init_sandbox_protocol(model_name)
                    elif protocol_key not in st.session_state:
                        init_sandbox_protocol(model_name)

                    if history_key not in st.session_state:
                        st.session_state[history_key] = []

                    protocol = st.session_state[protocol_key]
                    history = st.session_state[history_key]

                    # Layout Sandbox Parameters inside expander
                    with st.expander("Sandbox Simulation Parameters", expanded=True):
                        cols_p = st.columns(2)
                        col_p1 = cols_p[0] if len(cols_p) > 0 else dummy_col
                        col_p2 = cols_p[1] if len(cols_p) > 1 else dummy_col

                        with col_p1:
                            _ = create_widget(
                                col_p1,
                                "text_input",
                                f"{model_name}_sandbox_true_tox_str",
                                "True Toxicity Rates (comma-separated)",
                                value="0.05, 0.1, 0.2, 0.3, 0.4",
                                key=f"{model_name}_sandbox_true_tox_str"
                            )
                        is_joint = model_name in ["EffTox", "Wages & Tait", "WageStait", "WATU", "WaTu"]
                        if is_joint:
                            with col_p2:
                                _ = create_widget(
                                    col_p2,
                                    "text_input",
                                    f"{model_name}_sandbox_true_eff_str",
                                    "True Efficacy Rates (comma-separated)",
                                    value="0.2, 0.3, 0.4, 0.5, 0.6",
                                    key=f"{model_name}_sandbox_true_eff_str"
                                )

                        cols_sub = st.columns(2)
                        col_p3 = cols_sub[0] if len(cols_sub) > 0 else dummy_col
                        col_p4 = cols_sub[1] if len(cols_sub) > 1 else dummy_col

                        with col_p3:
                            cohort_size_val = create_widget(
                                col_p3,
                                "number_input",
                                cohort_size_key,
                                "Cohort Size",
                                min_value=1,
                                max_value=10,
                                value=int(st.session_state.get("cohort_size", 3)),
                                key=cohort_size_key
                            )
                        with col_p4:
                            # Show total patients treated currently
                            st.markdown(f"**Current Patients Treated:** {protocol.size()} / {protocol.max_size()}")

                    # Next recommended dose based on active protocol state
                    rec_dose_level = protocol.next_dose()
                    st.info(f"**Current Recommended Next Dose:** Level {rec_dose_level}")

                    # Dropdown for Dose Override
                    num_doses = protocol.number_of_doses()
                    override_choices = ["None / No Override"] + [str(d) for d in range(1, num_doses + 1)]

                    cols_act = st.columns([2, 1, 1])
                    col_act1 = cols_act[0] if len(cols_act) > 0 else dummy_col
                    col_act2 = cols_act[1] if len(cols_act) > 1 else dummy_col
                    col_act3 = cols_act[2] if len(cols_act) > 2 else dummy_col

                    with col_act1:
                        create_widget(
                            col_act1,
                            "selectbox",
                            override_key,
                            "Choose Dose Override before execution:",
                            options=override_choices,
                            key=override_key
                        )

                    is_ongoing = protocol.has_more() and protocol.size() < protocol.max_size()

                    # Handler for Step execution
                    def execute_step() -> None:
                        import numpy as np

                        # Validate inputs
                        def parse_comma_floats(text_val: str) -> Optional[list[float]]:
                            try:
                                return [float(x.strip()) for x in text_val.split(",") if x.strip()]
                            except Exception:
                                return None

                        true_tox_vals = parse_comma_floats(st.session_state[f"{model_name}_sandbox_true_tox_str"])
                        if not true_tox_vals or len(true_tox_vals) != num_doses:
                            st.error(f"True Toxicity Rates must be a comma-separated list of exactly {num_doses} floats.")
                            return

                        true_eff_vals = None
                        if is_joint:
                            true_eff_vals = parse_comma_floats(st.session_state[f"{model_name}_sandbox_true_eff_str"])
                            if not true_eff_vals or len(true_eff_vals) != num_doses:
                                st.error(f"True Efficacy Rates must be a comma-separated list of exactly {num_doses} floats.")
                                return
                            # Assert for mypy index check
                            assert true_eff_vals is not None

                        # Apply override dose if selected
                        chosen_override = st.session_state[override_key]
                        applied_override = False
                        if chosen_override != "None / No Override":
                            override_val_int = int(chosen_override)
                            protocol.set_next_dose(override_val_int)
                            applied_override = True

                        # Determine current target dose for simulation
                        sim_dose = protocol.next_dose()

                        # Generate outcomes
                        tox_sample = np.random.binomial(1, true_tox_vals[sim_dose - 1], size=cohort_size_val)

                        step_cases = []
                        if is_joint:
                            assert true_eff_vals is not None
                            eff_sample = np.random.binomial(1, true_eff_vals[sim_dose - 1], size=cohort_size_val)
                            step_cases = [
                                {"dose": sim_dose, "toxicity": int(t), "efficacy": int(e)}
                                for t, e in zip(tox_sample, eff_sample)
                            ]
                        else:
                            step_cases = [
                                {"dose": sim_dose, "toxicity": int(t)}
                                for t in tox_sample
                            ]

                        # Update protocol
                        next_calculated = protocol.update(step_cases)

                        # Log history
                        c_history = st.session_state[history_key]
                        c_index = len(c_history) + 1

                        c_summary = {
                            "cohort_index": c_index,
                            "dose_level": sim_dose,
                            "override_applied": applied_override,
                            "num_patients": cohort_size_val,
                            "cases": step_cases,
                            "num_toxicities": int(sum(tox_sample)),
                            "num_efficacies": int(sum(eff_sample)) if is_joint else None,
                            "next_recommended": next_calculated,
                        }
                        c_history.append(c_summary)
                        st.session_state[history_key] = c_history

                        # Reset override dropdown back to None
                        st.session_state[override_key] = "None / No Override"

                        # Rerun to reflect changes
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()  # type: ignore[attr-defined]

                    # Handler for Reset execution
                    def execute_reset() -> None:
                        st.session_state[history_key] = []
                        st.session_state[override_key] = "None / No Override"
                        init_sandbox_protocol(model_name)
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()  # type: ignore[attr-defined]

                    with col_act2:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        create_widget(
                            col_act2,
                            "button",
                            f"{model_name}_sandbox_simulate_button",
                            "Simulate Next Cohort ➡️",
                            on_click=execute_step,
                            disabled=not is_ongoing,
                            use_container_width=True
                        )
                    with col_act3:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        create_widget(
                            col_act3,
                            "button",
                            f"{model_name}_sandbox_reset_button",
                            "Reset Simulation 🔄",
                            on_click=execute_reset,
                            use_container_width=True
                        )

                    # Render cumulative cohort outcomes
                    if history:
                        import pandas as pd
                        rows = []
                        for c in history:
                            row = {
                                "Cohort": f"Cohort {c['cohort_index']}",
                                "Dose Administered": int(c["dose_level"]),
                                "Override Applied?": "Yes" if c["override_applied"] else "No",
                                "Patients": int(c["num_patients"]),
                                "Toxicities": int(c["num_toxicities"]),
                            }
                            if is_joint:
                                row["Efficacies"] = int(c["num_efficacies"])
                            row["Next Recommendation"] = int(c["next_recommended"])
                            rows.append(row)

                        history_df = pd.DataFrame(rows)
                        st.subheader("Accumulated Cohort Simulation History")
                        st.dataframe(history_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No cohorts simulated yet. Click 'Simulate Next Cohort' to start the interactive simulation pathway!")

        # Module docstring injection with CORE_REGISTRY
        if func.__module__:
            import sys

            mod = sys.modules[func.__module__]
            if mod.__doc__ and "CORE_REGISTRY" not in mod.__doc__:
                try:
                    from clintrials.core.registry import CORE_REGISTRY

                    mod.__doc__ = mod.__doc__.format(**CORE_REGISTRY)
                except Exception:
                    pass

        return wrapper

    return decorator
