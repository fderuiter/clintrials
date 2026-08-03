"""Random number generation utilities for reproducible simulations."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_rng(seed: Any = None) -> np.random.Generator:
    """Centralized RNG utility to create local generator objects for reproducibility."""
    return np.random.default_rng(seed)
