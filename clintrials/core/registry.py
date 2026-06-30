"""
Centralized registry for all statistical constants.
"""

from typing import Dict, Any

REGISTRY: Dict[str, Any] = {
    # Integration limits
    "gsd_maxpts": 1000000,
    "gsd_abseps": 1e-5,
    
    # Quadrature nodes
    "crm_deg": 40,
    
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

def inject_docs():
    """
    Decorator to inject registry constants into docstrings.
    """
    def decorator(obj):
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(**REGISTRY)
        return obj
    return decorator
