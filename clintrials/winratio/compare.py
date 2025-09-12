"""Compare two subjects component-wise in a hierarchical manner."""

from __future__ import annotations

from typing import Iterable


def compare_subjects(subject1: Iterable[int], subject2: Iterable[int]) -> str:
    """
    Compare two subjects across multiple components hierarchically.

    The first differing component determines the winner. Higher values are better.

    Args:
        subject1 (Iterable[int]): Outcomes for the first subject.
        subject2 (Iterable[int]): Outcomes for the second subject.

    Returns:
        str: 'win' if subject1 wins, 'loss' if subject1 loses, or 'tie'.
    """
    for i in range(len(subject1)):
        if subject1[i] > subject2[i]:
            return "win"
        if subject1[i] < subject2[i]:
            return "loss"
    return "tie"
