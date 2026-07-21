"""Framework tools for declarative and reusable simulation dashboard views."""

from functools import wraps

from clintrials.core.viz_interface import get_visualization_provider
from clintrials.visualization.dashboard.factory import render_accessible_chart


class BaseSimulationView:
    """Base declarative view class for trial design simulation dashboards."""

    model_name = ""
    title = ""
    file_prefix = ""
    param_space_config = {}
    model_class = None
    var_map = None
    csv_index = True
    skip_summary_table = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
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
                preview = cls.preview_sims

            PROTOCOL_REGISTRY.register_manual(
                cls.model_name, render_func=decorated_render, preview_func=preview
            )

    @classmethod
    def _base_render(cls, sims, ps=None):
        """Render the sidebar controls, parse parameter combinations, and execute the view mapping."""
        from clintrials.core.simulation import extract_sim_data

        func_map = cls.model_class.get_summary_functions()
        summary_df = extract_sim_data(
            sims, ps, func_map, var_map=cls.var_map, return_type="dataframe"
        )

        figures = cls.build_figures(summary_df)
        return summary_df, figures

    @classmethod
    def build_figures(cls, summary_df):
        """Build figures from the simulation summary dataframe. Should be overridden."""
        return []


def dashboard_view(title: str, model_name: str, file_prefix: str, csv_index: bool = True, skip_summary_table: bool = False, param_space_config: dict = None):  # type: ignore
    """Decorator to generate a standard dashboard view."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import streamlit as st

            if not hasattr(st, "fragment"):
                st.fragment = lambda f: f
            if not hasattr(st, "columns"):
                st.columns = lambda x: (st, st)

            ps = None
            if param_space_config is not None:
                st.sidebar.header("Trial Parameters")
                from clintrials.utils import ParameterSpace
                ps = ParameterSpace()
                for k, v in param_space_config.items():
                    ps.add(k, v)
                st.sidebar.json(param_space_config)

            st.header(title)

            try:
                if ps is not None:
                    kwargs["ps"] = ps
                result = func(*args, **kwargs)
                if result is None:
                    return

                # Check if it returned 2 or 3 items
                if isinstance(result, tuple) and len(result) == 3:
                    summary_df, figures, extra_text_summaries = result
                else:
                    summary_df, figures = result
                    extra_text_summaries = None

                if summary_df is not None and not summary_df.empty:
                    if not skip_summary_table:
                        st.subheader("Simulation Summary")
                        html_table = summary_df.to_html()
                        html_table = html_table.replace("<th></th>", "<th>Index</th>")
                        st.markdown(html_table, unsafe_allow_html=True)

                text_summaries = []
                if extra_text_summaries:
                    if isinstance(extra_text_summaries, list):
                        text_summaries.extend(extra_text_summaries)
                    else:
                        text_summaries.append(extra_text_summaries)

                if figures:
                    st.header("Operating Characteristics" if model_name != "Win Ratio" else "Visualizations")
                    for fig_title, fig in figures:
                        if fig_title:
                            st.subheader(fig_title)

                        meta = getattr(getattr(fig, "layout", None), "meta", "No data summary available.")
                        text_summaries.append(meta)
                        render_accessible_chart(st, fig)
                elif summary_df is not None and not summary_df.empty:
                    pass

                if summary_df is None or summary_df.empty:
                    return

                st.header("Export Results")
                col1, col2 = st.columns(2)

                csv_data = summary_df.to_csv(index=csv_index)
                getattr(col1, "download_button", lambda *args, **kwargs: None)(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{file_prefix}.csv",
                    mime="text/csv",
                )

                viz_provider = get_visualization_provider()
                pdf_data = viz_provider.generate_pdf_report(
                    summary_df, model_name, text_summaries=text_summaries
                ) if viz_provider else None

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
            except Exception as e:
                if hasattr(st, "error"):
                    st.error(f"An error occurred during summarization or plotting: {e}")
                else:
                    raise e

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
