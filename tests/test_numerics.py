import numpy as np
from scipy.stats import norm
from unittest.mock import patch

from clintrials.core.numerics import integrate_posterior_1d


def test_integrate_posterior_1d_expands():
    logpost = lambda x: norm(loc=3, scale=1).logpdf(x)
    val, diag = integrate_posterior_1d(
        logpost,
        lambda x: x,
        lo=-1,
        hi=1,
        return_diagnostics=True,
    )
    assert abs(val - 3) < 1e-2
    assert diag["expansions"] > 0


def test_integrate_posterior_1d_no_expand():
    logpost = lambda x: norm(loc=0, scale=1).logpdf(x)
    val, diag = integrate_posterior_1d(
        logpost,
        lambda x: x,
        lo=-5,
        hi=5,
        return_diagnostics=True,
    )
    assert abs(val) < 1e-2
    assert diag["expansions"] == 0


def test_integrate_posterior_1d_no_diag():
    logpost = lambda x: norm(loc=0, scale=1).logpdf(x)
    val = integrate_posterior_1d(
        logpost,
        lambda x: x,
        lo=-5,
        hi=5,
        return_diagnostics=False,
    )
    assert abs(val) < 1e-2


def test_integrate_posterior_1d_warn_on_max():
    logpost = lambda x: norm(loc=10, scale=1).logpdf(x)
    with np.testing.assert_warns(RuntimeWarning):
        integrate_posterior_1d(
            logpost,
            lambda x: x,
            lo=-1,
            hi=1,
            max_expansions=1,
            warn_on_max=True,
        )

def test_expansion_logic_is_correct():
    """
    Test that the expansion logic is correct.
    The new width should be width * (1 + expand_factor).
    """
    logpost = lambda x: norm(loc=10, scale=0.1).logpdf(x)
    # The initial range is small and far from the mode, so it will expand
    initial_lo, initial_hi = -1, 1
    initial_width = initial_hi - initial_lo
    expand_factor = 0.5

    # We mock linspace to capture the arguments of the second call
    # The first call is with the initial lo and hi.
    # The second call is with the expanded lo and hi.
    with patch("numpy.linspace") as mock_linspace:
        # Mocking the return of linspace to avoid errors in the rest of the function
        # A simple array is enough, but it needs to have more than 2 elements for the max_at_edge check
        mock_linspace.return_value = np.array([-1, 0, 1])
        integrate_posterior_1d(
            logpost,
            lambda x: x,
            lo=initial_lo,
            hi=initial_hi,
            expand_factor=expand_factor,
            max_expansions=1,
            n_points=3
        )
        # The second call to linspace will have the expanded limits
        # call_args_list[1] is the second call. `args` is a tuple of positional arguments.
        expanded_lo, expanded_hi, _ = mock_linspace.call_args_list[1].args
        expanded_width = expanded_hi - expanded_lo

        expected_width = initial_width * (1 + expand_factor)
        assert np.isclose(expanded_width, expected_width)
