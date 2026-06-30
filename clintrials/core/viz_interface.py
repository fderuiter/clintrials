"""Abstract interface for visualization providers."""

import abc
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class VisualizationResult:
    chart: Any
    metadata: Optional[Any] = None
    title: str = ""
    summary_text: str = ""

    def _repr_html_(self):
        """Rich HTML representation for Jupyter Notebooks."""
        html = []
        if hasattr(self.chart, "_repr_html_"):
            html.append(self.chart._repr_html_())
        elif hasattr(self.chart, "to_html"):
            html.append(self.chart.to_html(full_html=False, include_plotlyjs="cdn"))
        
        if self.metadata is not None and hasattr(self.metadata, "to_html"):
            html.append(f"<details><summary><b>Accessibility Metadata: {self.title}</b></summary>")
            html.append(self.metadata.to_html(index=False))
            html.append("</details>")
            
        return "\n".join(html) if html else None

class VisualizationProvider(abc.ABC):
    @abc.abstractmethod
    def plot_dose_finding_outcomes(self, trial, chart_title=None) -> VisualizationResult:
        """Plot dose finding outcomes."""
        pass

    @abc.abstractmethod
    def plot_crm_toxicity_probabilities(self, trial, chart_title=None) -> VisualizationResult:
        """Plot CRM toxicity probabilities."""
        pass

    @abc.abstractmethod
    def plot_efftox_utility_contours(
        self,
        metric,
        prob_eff=None,
        prob_tox=None,
        n=100,
        util_lower=-0.8,
        util_upper=0.8,
        util_delta=0.2,
        title="EffTox utility contours",
        custom_points_label="priors",
    ) -> VisualizationResult:
        """Plot EffTox utility contours."""
        pass

    @abc.abstractmethod
    def plot_efftox_density(
        self,
        data_func,
        trial,
        x_name="",
        plot_title="",
        include_doses=None,
        boot_samps=1000,
    ) -> VisualizationResult:
        """Plot EffTox density."""
        pass

    @abc.abstractmethod
    def plot_winratio_simulations(self, power, average_ci, title="Win Ratio Simulations") -> VisualizationResult:
        """Plot Win-Ratio simulations."""
        pass


_provider = None


def get_visualization_provider():
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


def set_visualization_provider(provider: VisualizationProvider):
    """Set the current visualization provider."""
    global _provider
    _provider = provider
