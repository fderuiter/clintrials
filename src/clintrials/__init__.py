"""A library of clinical trial designs and methods."""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

__all__ = [
    "dosefinding",
    "phase2",
    "coll",
    "common",
    "recruitment",
    "simulation",
    "stats",
    "tte",
    "util",
]

import logging

# Attach a NullHandler to avoid logging warnings on import
logging.getLogger(__name__).addHandler(logging.NullHandler())
