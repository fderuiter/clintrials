import numpy as np
import pytest

from clintrials.core.math import hyperbolic_tan, inverse_hyperbolic_tan


def test_hyperbolic_tan_basic():
    # When beta=0, exp(0) = 1, so the formula is (tanh(x) + 1) / 2
    # For x = 0, tanh(0) = 0 => (0 + 1) / 2 = 0.5
    assert np.isclose(hyperbolic_tan(0, beta=0), 0.5)

    # For x = 0.5, beta = 1, value is approx 0.4267599709486024
    val = hyperbolic_tan(0.5, beta=1)
    assert np.isclose(val, 0.4267599709486024)

def test_inverse_hyperbolic_tan_basic():
    # Inverse of hyperbolic_tan(0, beta=0) = 0.5
    assert np.isclose(inverse_hyperbolic_tan(0.5, beta=0), 0.0)

    # Inverse of 0.4267599709486024 with beta=1
    val = inverse_hyperbolic_tan(0.4267599709486024, beta=1)
    assert np.isclose(val, 0.5)

@pytest.mark.parametrize("x", [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [-1.0, 0.0, 1.0, 2.0])
def test_hyperbolic_tan_consistency(x, beta):
    # Test forward and backward mapping
    # hyperbolic_tan(x, beta) -> y
    # inverse_hyperbolic_tan(y, beta) -> x
    y = hyperbolic_tan(x, beta=beta)

    # Check bounds
    assert 0.0 < y < 1.0

    # Check consistency
    x_approx = inverse_hyperbolic_tan(y, beta=beta)
    assert np.isclose(x, x_approx, atol=1e-7)

def test_hyperbolic_tan_a0_ignored():
    # a0 parameter exists for call signature matching but is ignored
    assert hyperbolic_tan(0.5, a0=100, beta=1) == hyperbolic_tan(0.5, a0=0, beta=1)
    assert inverse_hyperbolic_tan(0.5, a0=100, beta=1) == inverse_hyperbolic_tan(0.5, a0=0, beta=1)
