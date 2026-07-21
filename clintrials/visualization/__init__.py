from __future__ import annotations
"""Visualization namespace. Ensure you have installed the optional viz dependencies."""

from clintrials.visualization.provider import (
    create_bar_chart,
    create_line_chart,
    generate_text_summary,
    plot_bivariate_simulation_recommendation,
    plot_crm_simulation_recommendation,
    plot_crm_toxicity_probabilities,
    plot_dose_finding_outcomes,
    plot_efftox_density,
    plot_efftox_simulation_acceptability,
    plot_efftox_utility_contours,
    plot_winratio_power_curve,
)

__all__ = [
    "create_bar_chart",
    "create_line_chart",
    "generate_text_summary",
    "plot_dose_finding_outcomes",
    "plot_crm_toxicity_probabilities",
    "plot_efftox_utility_contours",
    "plot_efftox_density",
    "plot_crm_simulation_recommendation",
    "plot_bivariate_simulation_recommendation",
    "plot_efftox_simulation_acceptability",
    "plot_winratio_power_curve",
]
