import numpy as np

from clintrials.winratio.data_generation import generate_data


def test_winratio_data_generation_reproducibility():
    from clintrials.core.rng import get_rng

    rng1 = get_rng(42)
    rng2 = get_rng(42)
    rng3 = get_rng(43)

    a1, b1 = generate_data(10, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, rng=rng1)
    a2, b2 = generate_data(10, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, rng=rng2)
    a3, b3 = generate_data(10, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, rng=rng3)

    # Same seed -> identical data
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)

    # Different seeds -> different data
    assert not (np.array_equal(a1, a3) and np.array_equal(b1, b3))
