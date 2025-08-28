import numpy as np
from scipy.stats import norm

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
