from __future__ import annotations
import numpy as np


def get_rng(seed=None):  # type: ignore
    """Centralized RNG utility to create local generator objects for reproducibility."""
    return np.random.default_rng(seed)
