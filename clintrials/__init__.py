__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

from . import core
from . import dosefinding
from . import phase2
from . import utils
from .core.math import fgm_joint_prob
from .core.numerics import adaptive_mc_integration, integrate_posterior_1d

__all__ = [
    "core",
    "dosefinding",
    "phase2",
    "utils",
    "fgm_joint_prob",
    "adaptive_mc_integration",
    "integrate_posterior_1d",
]

import logging

# Attach a NullHandler to avoid logging warnings on import
logging.getLogger(__name__).addHandler(logging.NullHandler())
