"""Win-ratio simulation utilities."""

from .compare import compare_subjects
from .data_generation import generate_data
from .main import run_simulation
from .simulate import simulate_comparisons
from .statistics import (
    calculate_confidence_intervals,
    calculate_p_value,
    calculate_win_ratio,
)

__all__ = [
    "compare_subjects",
    "generate_data",
    "simulate_comparisons",
    "calculate_confidence_intervals",
    "calculate_p_value",
    "calculate_win_ratio",
    "run_simulation",
]
