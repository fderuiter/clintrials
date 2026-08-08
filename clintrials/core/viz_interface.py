# SPDX-License-Identifier: MIT

"""Abstract interface for visualization providers.

Random Seed Strategy: {viz_interface_seed_strategy}
"""

from __future__ import annotations

import abc
from typing import Any


class VisualizationProvider(abc.ABC):
    """Abstract base class representing a visualization rendering provider."""

    @abc.abstractmethod
    def plot_dose_finding_outcomes(self, trial, chart_title=None):  # type: ignore
        """Plot dose finding outcomes."""
        pass

    @abc.abstractmethod
    def plot_crm_toxicity_probabilities(self, trial, chart_title=None):  # type: ignore
        """Plot CRM toxicity probabilities."""
        pass

    @abc.abstractmethod
    def generate_pdf_report(self, df, design_type, text_summaries=None):  # type: ignore
        """Generates an accessibility-first PDF report for trial simulations."""
        pass

    def plot_crm_simulation_recommendation(self, summary_df, high_contrast=False):  # type: ignore
        """Plots CRM simulation recommendation probabilities."""
        from clintrials.visualization.provider import get_default_provider

        return get_default_provider().plot_crm_simulation_recommendation(
            summary_df, high_contrast=high_contrast
        )

    def plot_bivariate_simulation_recommendation(self, summary_df, high_contrast=False):  # type: ignore
        """Plots EffTox simulation recommendation probabilities."""
        from clintrials.visualization.provider import get_default_provider

        return get_default_provider().plot_bivariate_simulation_recommendation(
            summary_df, high_contrast=high_contrast
        )

    def plot_efftox_simulation_acceptability(self, summary_df, high_contrast=False):  # type: ignore
        """Plots EffTox simulation acceptability probabilities."""
        from clintrials.visualization.provider import get_default_provider

        return get_default_provider().plot_efftox_simulation_acceptability(
            summary_df, high_contrast=high_contrast
        )

    def plot_winratio_power_curve(self, df, high_contrast=False):  # type: ignore
        """Plots a Win Ratio simulation power curve."""
        from clintrials.visualization.provider import get_default_provider

        return get_default_provider().plot_winratio_power_curve(
            df, high_contrast=high_contrast
        )

    def create_bar_chart(  # type: ignore
        self, df, x, y, color, title, labels=None, high_contrast=False
    ):
        """Creates a centralized bar chart with accessibility standards."""
        from clintrials.visualization.provider import get_default_provider

        return get_default_provider().create_bar_chart(
            df, x, y, color, title, labels=labels, high_contrast=high_contrast
        )


_provider = None


def get_visualization_provider() -> Any:
    """Get the current visualization provider."""
    global _provider
    if _provider is None:
        try:
            from clintrials.visualization.provider import get_default_provider

            _provider = get_default_provider()
        except ImportError:
            raise ImportError(
                "Visualization libraries are not installed. Install with `pip install clintrials[viz]` "
                "or register a custom provider using `set_visualization_provider`."
            )
    return _provider


def set_visualization_provider(provider: VisualizationProvider) -> None:
    """Set the current visualization provider."""
    global _provider
    _provider = provider


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY

    __doc__ = __doc__.format(**CORE_REGISTRY)
