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
