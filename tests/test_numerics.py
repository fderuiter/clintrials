import numpy as np
from scipy.stats import norm

from clintrials.numerics import integrate_posterior_1d


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
