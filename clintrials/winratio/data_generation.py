# SPDX-License-Identifier: MIT

"""Generate synthetic data for win-ratio simulations.

Random Seed Strategy: {data_generation_seed_strategy}
"""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_data(  # type: ignore
    num_subjects_A: int,
    num_subjects_B: int,
    p_y1_A: float,
    p_y1_B: float,
    p_y2_A: float,
    p_y2_B: float,
    p_y3_A: float,
    p_y3_B: float,
    rng: Any = None,
):
    """Generate data for treatment (A) and control (B) groups.

    Each subject has three binary outcomes (y1, y2, y3).

    Args:
        num_subjects_A (int): Number of subjects in Group A.
        num_subjects_B (int): Number of subjects in Group B.
        p_y1_A (float): Probability of outcome ``y1`` equals 1 for Group A.
        p_y1_B (float): Probability of outcome ``y1`` equals 1 for Group B.
        p_y2_A (float): Probability of outcome ``y2`` equals 1 for Group A.
        p_y2_B (float): Probability of outcome ``y2`` equals 1 for Group B.
        p_y3_A (float): Probability of outcome ``y3`` equals 1 for Group A.
        p_y3_B (float): Probability of outcome ``y3`` equals 1 for Group B.
        rng (Any, optional): Local random number generator instance.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Two arrays representing the
            subjects in Groups A and B respectively.
    """
    if rng is None:
        from clintrials.core.rng import get_rng
        rng = get_rng()

    group_A = np.vstack(
        [
            rng.binomial(1, p_y1_A, num_subjects_A),
            rng.binomial(1, p_y2_A, num_subjects_A),
            rng.binomial(1, p_y3_A, num_subjects_A),
        ]
    ).T

    group_B = np.vstack(
        [
            rng.binomial(1, p_y1_B, num_subjects_B),
            rng.binomial(1, p_y2_B, num_subjects_B),
            rng.binomial(1, p_y3_B, num_subjects_B),
        ]
    ).T

    return group_A, group_B


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
