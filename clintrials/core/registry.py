"""Centralized registry for all statistical constants."""

from __future__ import annotations

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
from typing import TYPE_CHECKING, Callable, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

class ProtocolRegistry:
    """A registry for clinical trial protocol designs and their visualization methods."""

    def __init__(self) -> None:
        """Initializes a new ProtocolRegistry instance."""
        self._designs: Dict[str, Dict[str, Any]] = {}
        self._discovered: bool = False
        self._snapshot: Optional[Dict[str, Dict[str, Any]]] = None
        self._discovered_snapshot: Optional[bool] = None

    def snapshot(self) -> None:
        """Capture a snapshot of the current registry state."""
        self._snapshot = {k: v.copy() for k, v in self._designs.items()}
        self._discovered_snapshot = self._discovered

    def restore(self) -> None:
        """Restore the registry state from the snapshot."""
        if self._snapshot is not None:
            self._designs = {k: v.copy() for k, v in self._snapshot.items()}
        if self._discovered_snapshot is not None:
            self._discovered = self._discovered_snapshot

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        try:
            import clintrials.visualization.dashboard.views as views
            if views.__path__:
                for _, name, _ in pkgutil.iter_modules(views.__path__):
                    if name.endswith("_view"):
                        importlib.import_module(f"clintrials.visualization.dashboard.views.{name}")
        except Exception:
            pass

    def register(self, name: str, preview_func: Optional[Callable] = None) -> Callable:  # type: ignore
        """Register a protocol design with an optional preview simulation function.

        Args:
            name (str): The name of the protocol design.
            preview_func (callable, optional): A function to generate a preview simulation.
                Defaults to None.

        Returns:
            callable: A decorator function for registering the render method.
        """
        def decorator(render_func: Callable) -> Callable:  # type: ignore
            if name in self._designs:
                logger.warning(f"Duplicate registration encountered for design name: {name}")
            else:
                self._designs[name] = {}
            self._designs[name]["render"] = render_func
            if preview_func:
                self._designs[name]["preview"] = preview_func
            return render_func
        return decorator

    def register_manual(self, name: str, render_func: Callable, preview_func: Optional[Callable] = None) -> None:  # type: ignore
        """Manually register a protocol design with its render and preview functions."""
        if name in self._designs:
            logger.warning(f"Duplicate manual registration encountered for design name: {name}")
        self._designs[name] = {"render": render_func, "preview": preview_func}

    def get_designs(self) -> List[str]:
        """Get a list of all registered protocol design names."""
        self._discover()
        return list(self._designs.keys())

    def get_render(self, name: str) -> Optional[Callable]:  # type: ignore
        """Get the render function for a registered protocol design."""
        self._discover()
        return self._designs.get(name, {}).get("render")

    def get_preview(self, name: str) -> Optional[Callable]:  # type: ignore
        """Get the preview function for a registered protocol design."""
        self._discover()
        return self._designs.get(name, {}).get("preview")

PROTOCOL_REGISTRY = ProtocolRegistry()

def inject_docs() -> Callable:  # type: ignore
    """Decorator to inject registry constants into docstrings."""
    def decorator(obj: Any) -> Any:
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(**CORE_REGISTRY)
        return obj
    return decorator

