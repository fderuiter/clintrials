# SPDX-License-Identifier: MIT

# type: ignore
from clintrials.core.viz_interface import (
    VisualizationProvider,
    get_visualization_provider,
    set_visualization_provider,
)


class DummyVisualizationProvider(VisualizationProvider):
    def plot_dose_finding_outcomes(self, trial, chart_title=None):
        return "dose_outcomes"

    def plot_crm_toxicity_probabilities(self, trial, chart_title=None):
        return "crm_probs"

    def generate_pdf_report(self, df, design_type, text_summaries=None):
        return "pdf_report"


def test_visualization_provider_registration():
    """Test setting and getting a custom visualization provider."""
    provider = DummyVisualizationProvider()
    set_visualization_provider(provider)
    current = get_visualization_provider()
    assert current is provider
    assert current.plot_dose_finding_outcomes(None) == "dose_outcomes"


class PartialCustomProvider(VisualizationProvider):
    def plot_dose_finding_outcomes(self, trial, chart_title=None):
        return "custom_dose_outcomes"

    def plot_crm_toxicity_probabilities(self, trial, chart_title=None):
        return "custom_crm_probs"

    def generate_pdf_report(self, df, design_type, text_summaries=None):
        return "custom_pdf_report"

    def plot_crm_simulation_recommendation(self, summary_df, high_contrast=False):
        return "custom_crm_rec"

    def plot_winratio_power_curve(self, df, high_contrast=False):
        return "custom_winratio_power"


def test_custom_provider_selective_override_and_fallback(monkeypatch):
    """Test that custom providers can override a subset and fall back to defaults for others."""
    old_provider = get_visualization_provider()
    try:
        provider = PartialCustomProvider()
        set_visualization_provider(provider)

        current = get_visualization_provider()
        assert current is provider

        # Overridden methods should return custom results
        assert current.plot_crm_simulation_recommendation(None) == "custom_crm_rec"
        assert current.plot_winratio_power_curve(None) == "custom_winratio_power"

        # Non-overridden methods should fallback to the default provider
        # Mock default provider methods to verify routing
        import clintrials.visualization as viz
        monkeypatch.setattr(viz, "plot_bivariate_simulation_recommendation", lambda df, high_contrast=False: "default_bivariate_rec")
        monkeypatch.setattr(viz, "plot_efftox_simulation_acceptability", lambda df, high_contrast=False: "default_efftox_accept")
        monkeypatch.setattr(viz, "create_bar_chart", lambda df, x, y, color, title, labels=None, high_contrast=False: "default_bar_chart")

        assert current.plot_bivariate_simulation_recommendation(None) == "default_bivariate_rec"
        assert current.plot_efftox_simulation_acceptability(None) == "default_efftox_accept"
        assert current.create_bar_chart(None, None, None, None, None) == "default_bar_chart"
    finally:
        set_visualization_provider(old_provider)


def test_default_provider_routing_without_custom_provider():
    """Test that default plotting provider works correctly when no custom provider is registered."""
    old_provider = get_visualization_provider()
    try:
        from clintrials.visualization.provider import get_default_provider
        set_visualization_provider(get_default_provider())

        current = get_visualization_provider()
        from clintrials.visualization.provider import DefaultVisualizationProvider
        assert isinstance(current, DefaultVisualizationProvider)
    finally:
        set_visualization_provider(old_provider)
