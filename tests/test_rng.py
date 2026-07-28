# type: ignore
import numpy as np

from clintrials.core.rng import get_rng


def test_get_rng():
    """Test get_rng utility creates a valid numpy Generator."""
    rng = get_rng(42)
    assert isinstance(rng, np.random.Generator)
    # Check reproducible generation
    val1 = rng.integers(0, 100)
    rng2 = get_rng(42)
    val2 = rng2.integers(0, 100)
    assert val1 == val2
