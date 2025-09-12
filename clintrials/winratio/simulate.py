"""Pairwise win-ratio comparisons between treatment and control subjects."""

from __future__ import annotations

from typing import Dict

from .compare import compare_subjects


def simulate_comparisons(treatment_group, control_group) -> dict[str, int]:
    """
    Compare every treatment subject with every control subject.

    Args:
        treatment_group (numpy.ndarray): 2D array of subjects in the
            treatment group.
        control_group (numpy.ndarray): 2D array of subjects in the control
            group.

    Returns:
        dict[str, int]: Counts of wins, losses and ties for the treatment
            group.
    """
    results = {"wins": 0, "losses": 0, "ties": 0}
    for treatment_subj in treatment_group:
        for control_subj in control_group:
            result = compare_subjects(treatment_subj, control_subj)
            if result == "win":
                results["wins"] += 1
            elif result == "loss":
                results["losses"] += 1
            else:
                results["ties"] += 1
    return results
