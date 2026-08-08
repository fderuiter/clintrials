# SPDX-License-Identifier: MIT

"""Compare two subjects component-wise in a hierarchical manner.

Random Seed Strategy: {compare_seed_strategy}
"""

from __future__ import annotations

from typing import Iterable


def compare_subjects(subject1: Iterable[int], subject2: Iterable[int]) -> str:
    """Compare two subjects across multiple components hierarchically.

    The first differing component determines the winner. Higher values are better.

    Args:
        subject1 (Iterable[int]): Outcomes for the first subject.
        subject2 (Iterable[int]): Outcomes for the second subject.

    Returns:
        str: 'win' if subject1 wins, 'loss' if subject1 loses, or 'tie'.
    """
    import numpy as np

    from clintrials.core.errors import ErrorTemplates

    if not isinstance(subject1, (list, tuple, np.ndarray)) or not isinstance(
        subject2, (list, tuple, np.ndarray)
    ):
        raise TypeError(
            "Subject outcomes must be sequence types (list, tuple, or numpy array)."
        )

    s1_arr = np.asarray(subject1)
    s2_arr = np.asarray(subject2)
    if s1_arr.ndim != 1 or s2_arr.ndim != 1:
        raise ValueError("Subject outcomes must be 1D sequences.")

    if s1_arr.shape != s2_arr.shape:
        raise ValueError(
            ErrorTemplates.MATCHING_LENGTHS.format(
                first_name="subject1", name="subject2"
            )
        )

    for i in range(len(subject1)):
        if subject1[i] > subject2[i]:
            return "win"
        if subject1[i] < subject2[i]:
            return "loss"
    return "tie"


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY

    __doc__ = __doc__.format(**CORE_REGISTRY)
