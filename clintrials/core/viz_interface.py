"""Abstract interface for visualization providers."""

import abc


class VisualizationProvider(abc.ABC):
    @abc.abstractmethod
    def plot_dose_finding_outcomes(self, trial, chart_title=None):
        pass

    @abc.abstractmethod
    def plot_crm_toxicity_probabilities(self, trial, chart_title=None):
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
    ):
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
    ):
        pass


_provider = None


def get_visualization_provider():
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
    global _provider
    _provider = provider
