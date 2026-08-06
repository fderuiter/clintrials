# SPDX-License-Identifier: MIT

"""Random number generation utilities for reproducible simulations."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_rng(seed: Any = None) -> np.random.Generator:
    """Centralized RNG utility to create local generator objects for reproducibility.

    Args:
        seed (Any, optional): The seed to initialize the Generator with. Can be an integer,
            an array of integers, a SeedSequence, or None. Defaults to None.

    Returns:
        numpy.random.Generator: The initialized random number generator.
    """
    return np.random.default_rng(seed)
