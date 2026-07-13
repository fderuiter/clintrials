from functools import wraps

from clintrials.core.viz_interface import get_visualization_provider
from clintrials.visualization.dashboard.factory import render_accessible_chart


def dashboard_view(title: str, model_name: str, file_prefix: str, csv_index: bool = True, skip_summary_table: bool = False):
    """Decorator to generate a standard dashboard view."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import streamlit as st
            
            if not hasattr(st, "fragment"):
                st.fragment = lambda f: f
            if not hasattr(st, "columns"):
                st.columns = lambda x: (st, st)

            st.header(title)

            try:
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

        # Module docstring injection with REGISTRY
        if func.__module__:
            import sys
            mod = sys.modules[func.__module__]
            if mod.__doc__ and "REGISTRY" not in mod.__doc__:
                try:
                    from clintrials.core.registry import REGISTRY
                    mod.__doc__ = mod.__doc__.format(**REGISTRY)
                except Exception:
                    pass

        return wrapper
    return decorator
