# SPDX-License-Identifier: MIT

"""Renders the Win Ratio simulation view in the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from clintrials.core.schema import WinRatioSchema
from clintrials.visualization.dashboard.factory import create_widget, render_metric
from clintrials.visualization.dashboard.views.framework import BaseSimulationView
from clintrials.winratio.main import run_winratio_simulations


class WinRatioView(BaseSimulationView):
    """View class for the Win Ratio simulation."""

    model_name = "Win Ratio"
    title = "Win Ratio Simulation"
    file_prefix = "winratio_simulation"
    csv_index = False
    skip_summary_table = True
    param_space_config = {}  # type: ignore

    @classmethod
    def _base_render(cls, sims=None, ps=None):  # type: ignore
        """Render the Win Ratio simulation interface."""
        # Use schema to generate UI inputs
        kwargs = {}
        for name, field in WinRatioSchema.model_fields.items():
            min_val = 0.0 if "Probability" in str(field.annotation) else 1
            max_val = 1.0 if "Probability" in str(field.annotation) else None

            kwargs[name] = create_widget(  # type: ignore
                st,
                "number_input",
                name,
                field.description,
                min_value=min_val,
                max_value=max_val,
                value=field.default,
            )

        if create_widget(st, "button", "run_simulation_button", "Run Simulation"):  # type: ignore
            from clintrials.visualization.dashboard.utils import announce_status_locally

            announce_status_locally("Simulation in progress", key="winratio-start")
            try:
                with st.spinner("Running simulation..."):
                    summary = run_winratio_simulations(**kwargs)
                    power = summary["power"]
                    average_ci = summary["average_ci"]
                announce_status_locally("Simulation completed", key="winratio-complete")
                st.success("Simulation complete")
            except Exception as e:
                announce_status_locally("Simulation failed", key="winratio-fail")
                raise e

            st.subheader("Results")

            if not hasattr(st, "columns"):
                st.columns = lambda x: (st, st)  # type: ignore[assignment, misc]
            met_col1, met_col2 = st.columns(2)

            render_metric(met_col1, "Power", power)  # type: ignore
            render_metric(met_col2, "Average 95% Confidence Interval", average_ci)  # type: ignore

            # Create simple DataFrame for export
            results_dict = kwargs.copy()
            results_dict["power"] = power
            results_dict["ci_lower"] = average_ci[0]
            results_dict["ci_upper"] = average_ci[1]

            df = pd.DataFrame([results_dict])

            from clintrials.core.viz_interface import get_visualization_provider

            fig = get_visualization_provider().plot_winratio_power_curve(  # type: ignore
                df, high_contrast=False
            )
            figures = [(None, fig)]

            from clintrials.visualization.helpers import format_number

            extra_text_summaries = [
                f"Power: {format_number(power)}\n95% CI: ({format_number(average_ci[0])}, {format_number(average_ci[1])})"
            ]

            return df, figures, extra_text_summaries

        return None


def render(*args: Any, **kwargs: Any) -> Any:
    """Module-level wrapper for backward compatibility and testing."""
    from clintrials.core.registry import PROTOCOL_REGISTRY

    render_func = PROTOCOL_REGISTRY.get_render("Win Ratio")
    if render_func:
        return render_func(*args, **kwargs)
