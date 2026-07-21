from __future__ import annotations
__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

from . import core, dosefinding, utils
from .core.math import fgm_joint_prob
from .core.numerics import (
    adaptive_mc_integration,
    integrate_posterior_1d,
    integrate_posterior_1d_adaptive,
    integrate_posterior_1d_nonadaptive,
)
from .dosefinding.watu import WATU

__all__ = [
    "core",
    "dosefinding",
    "utils",
    "fgm_joint_prob",
    "adaptive_mc_integration",
    "integrate_posterior_1d",
    "integrate_posterior_1d_adaptive",
    "integrate_posterior_1d_nonadaptive",
    "WATU",
]

import logging

# Attach a NullHandler to avoid logging warnings on import
logging.getLogger(__name__).addHandler(logging.NullHandler())
