"""Pairwise win-ratio comparisons between treatment and control subjects.

Random Seed Strategy: {simulate_seed_strategy}
"""

from __future__ import annotations

import numpy as np


def simulate_comparisons(treatment_group, control_group) -> dict[str, int]:  # type: ignore
    """Compare every treatment subject with every control subject.

    Args:
        treatment_group (numpy.ndarray): 2D array of subjects in the
            treatment group.
        control_group (numpy.ndarray): 2D array of subjects in the control
            group.

    Returns:
        dict[str, int]: Counts of wins, losses and ties for the treatment
            group.
    """
    t_group = np.asarray(treatment_group)
    c_group = np.asarray(control_group)

    if t_group.size == 0 or c_group.size == 0:
        return {"wins": 0, "losses": 0, "ties": 0}

    diff = t_group[:, np.newaxis, :] - c_group[np.newaxis, :, :]

    non_zero = diff != 0
    has_non_zero = non_zero.any(axis=2)

    first_non_zero_idx = np.argmax(non_zero, axis=2)

    first_diff = np.take_along_axis(diff, first_non_zero_idx[:, :, np.newaxis], axis=2).squeeze(axis=2)

    wins = np.sum((first_diff > 0) & has_non_zero)
    losses = np.sum((first_diff < 0) & has_non_zero)
    ties = np.sum(~has_non_zero)

    return {"wins": int(wins), "losses": int(losses), "ties": int(ties)}



# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
