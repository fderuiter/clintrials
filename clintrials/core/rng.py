import numpy as np

def get_rng(seed=None):
    """Centralized RNG utility to create local generator objects for reproducibility."""
    return np.random.default_rng(seed)
