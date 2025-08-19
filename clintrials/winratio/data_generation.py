"""Generate synthetic data for win-ratio simulations."""

from __future__ import annotations

import numpy as np


def generate_data(
    num_subjects_A: int,
    num_subjects_B: int,
    p_y1_A: float,
    p_y1_B: float,
    p_y2_A: float,
    p_y2_B: float,
    p_y3_A: float,
    p_y3_B: float,
):
    """Generate data for treatment (A) and control (B) groups.

    Each subject has three binary outcomes (y1, y2, y3).

    Args:
        num_subjects_A: Number of subjects in Group A.
        num_subjects_B: Number of subjects in Group B.
        p_y1_A: Probability of outcome ``y1`` equals 1 for Group A.
        p_y1_B: Probability of outcome ``y1`` equals 1 for Group B.
        p_y2_A: Probability of outcome ``y2`` equals 1 for Group A.
        p_y2_B: Probability of outcome ``y2`` equals 1 for Group B.
        p_y3_A: Probability of outcome ``y3`` equals 1 for Group A.
        p_y3_B: Probability of outcome ``y3`` equals 1 for Group B.

    Returns:
        Two arrays representing the subjects in Groups A and B respectively.
    """
    group_A = np.vstack(
        [
            np.random.binomial(1, p_y1_A, num_subjects_A),
            np.random.binomial(1, p_y2_A, num_subjects_A),
            np.random.binomial(1, p_y3_A, num_subjects_A),
        ]
    ).T

    group_B = np.vstack(
        [
            np.random.binomial(1, p_y1_B, num_subjects_B),
            np.random.binomial(1, p_y2_B, num_subjects_B),
            np.random.binomial(1, p_y3_B, num_subjects_B),
        ]
    ).T

    return group_A, group_B
