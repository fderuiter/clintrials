"""Centralized registry for all statistical constants."""

from typing import Any, Dict

CORE_REGISTRY: Dict[str, Any] = {
    # Integration limits
    "gsd_maxpts": 1000000,
    "gsd_abseps": 1e-5,

    # Quadrature nodes
    "crm_deg": 40,

    # CRM Integration parameters
    "crm_min_beta": -10,
    "crm_max_beta": 10,
    "crm_n_points": 2001,
    "crm_sample_size": 1000000,

    # Search intervals
    "gsd_brentq_first_min": -5,
    "gsd_brentq_first_max": 15,
    "gsd_brentq_second_min": -50,
    "gsd_brentq_second_max": 50,

    # Math constants (beta clipping)
    "math_clip_beta_min": -10,
    "math_clip_beta_max": 10,

    # Random seeds
    "gsd_multivariate_normal_seed": 42,
    "gsd_seed_strategy": "fixed seed (42)",
    "crm_seed_strategy": "global state",
    "math_seed_strategy": "not applicable",
    "validation_seed_strategy": "not applicable",
    "utils_seed_strategy": "not applicable",
    "unified_seed_strategy": "not applicable",
    "simulation_seed_strategy": "not applicable",
    "tte_seed_strategy": "not applicable",
    "schema_seed_strategy": "not applicable",
    "viz_interface_seed_strategy": "not applicable",
    "numerics_seed_strategy": "global state",
    "recruitment_seed_strategy": "not applicable",
    "stats_seed_strategy": "not applicable",
    "rng_seed_strategy": "not applicable",
    "report_seed_strategy": "not applicable",
    "protocol_seed_strategy": "not applicable",
    "provider_seed_strategy": "not applicable",
    "factory_seed_strategy": "not applicable",
    "main_seed_strategy": "not applicable",
    "winratio_view_seed_strategy": "not applicable",
    "crm_view_seed_strategy": "not applicable",
    "efftox_view_seed_strategy": "not applicable",
    "simulate_seed_strategy": "global state",
    "data_generation_seed_strategy": "not applicable",
    "compare_seed_strategy": "not applicable",
    "statistics_seed_strategy": "not applicable",
    "simple_seed_strategy": "not applicable",
    "peps2v1_seed_strategy": "not applicable",
    "peps2v2_seed_strategy": "not applicable",
    "efftox_seed_strategy": "global state",
    "efficacytoxicity_seed_strategy": "not applicable",
    "watu_seed_strategy": "not applicable",
    "wagestait_seed_strategy": "not applicable",

}

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


class ProtocolRegistry:
    """A registry for clinical trial protocol designs and their visualization methods."""

    def __init__(self):
        """Initializes a new ProtocolRegistry instance."""
        self._designs = {}
        self._discovered = False

    def _discover(self):
        if self._discovered:
            return
        self._discovered = True
        try:
            import clintrials.visualization.dashboard.views as views
            for _, name, _ in pkgutil.iter_modules(views.__path__):
                if name.endswith("_view"):
                    importlib.import_module(f"clintrials.visualization.dashboard.views.{name}")
        except Exception:
            pass

    def register(self, name, preview_func=None):
        """Register a protocol design with an optional preview simulation function.

        Args:
            name (str): The name of the protocol design.
            preview_func (callable, optional): A function to generate a preview simulation.
                Defaults to None.

        Returns:
            callable: A decorator function for registering the render method.
        """
        def decorator(render_func):
            if name in self._designs:
                logger.warning(f"Duplicate registration encountered for design name: {name}")
            else:
                self._designs[name] = {}
            self._designs[name]["render"] = render_func
            if preview_func:
                self._designs[name]["preview"] = preview_func
            return render_func
        return decorator

    def register_manual(self, name, render_func, preview_func=None):
        """Manually register a protocol design with its render and preview functions.

        Args:
            name (str): The name of the protocol design.
            render_func (callable): The rendering function for the design.
            preview_func (callable, optional): A function to generate a preview simulation.
                Defaults to None.

        Returns:
            None
        """
        if name in self._designs:
            logger.warning(f"Duplicate manual registration encountered for design name: {name}")
        self._designs[name] = {"render": render_func, "preview": preview_func}

    def get_designs(self):
        """Get a list of all registered protocol design names.

        Returns:
            list: A list containing the names of registered protocol designs.
        """
        self._discover()
        return list(self._designs.keys())

    def get_render(self, name):
        """Get the render function for a registered protocol design.

        Args:
            name (str): The name of the registered protocol design.

        Returns:
            callable or None: The render function if registered, otherwise None.
        """
        self._discover()
        return self._designs.get(name, {}).get("render")

    def get_preview(self, name):
        """Get the preview function for a registered protocol design.

        Args:
            name (str): The name of the registered protocol design.

        Returns:
            callable or None: The preview function if registered, otherwise None.
        """
        self._discover()
        return self._designs.get(name, {}).get("preview")

PROTOCOL_REGISTRY = ProtocolRegistry()

def inject_docs():
    """Decorator to inject registry constants into docstrings.

    Returns:
        callable: A decorator function that formats the docstring of the wrapped object.
    """
    def decorator(obj):
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(**CORE_REGISTRY)
        return obj
    return decorator
