import numpy as np
from scipy.stats import norm

from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU


def test_prob_eff_exceeds_backends_consistency():
    tox_prior = [0.1, 0.2, 0.3]
    tox_target = 0.3
    tox_limit = 0.33
    eff_limit = 0.05
    skeletons = [[0.6, 0.4, 0.2], [0.2, 0.4, 0.6]]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_limit,
        eff_limit,
        metric,
        first_dose=1,
        max_size=20,
    )

    # Add some data to get a non-trivial posterior
    cases = [(1, 0, 1), (1, 0, 1), (2, 0, 0), (3, 1, 0)]
    trial.update(cases)

    eff_cutoff = 0.3

    p_analytic = trial.prob_eff_exceeds(eff_cutoff, backend="analytic")
    p_mc = trial.prob_eff_exceeds(eff_cutoff, backend="mc", n=10**6)
    p_quad = trial.prob_eff_exceeds(eff_cutoff, backend="quadrature")

    # Quad should be very accurate. MC with 10^6 should be accurate to ~0.001
    assert np.allclose(p_quad, p_mc, atol=0.01)

    # Analytic (Laplace) might be slightly different but should be in the same ballpark
    assert np.allclose(p_quad, p_analytic, atol=0.05)


def test_prob_eff_exceeds_edge_cases():
    tox_prior = [0.1, 0.2]
    tox_target = 0.3
    tox_limit = 0.33
    eff_limit = 0.05
    skeletons = [[0.6, 0.4]]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_limit,
        eff_limit,
        metric,
        first_dose=1,
        max_size=20,
    )

    # eff_cutoff >= 1
    p = trial.prob_eff_exceeds(1.0)
    assert np.all(p == 0.0)

    # eff_cutoff < 0
    p = trial.prob_eff_exceeds(-0.1)
    assert np.all(p == 1.0)

    # Skeleton with 0 or 1
    trial.skeletons = [[1.0, 0.5, 0.0]]
    trial.most_likely_model_index = 0
    p = trial.prob_eff_exceeds(0.5, backend="quadrature")
    assert p[0] == 1.0
    assert p[2] == 0.0
    assert 0.0 < p[1] < 1.0


def test_prob_eff_exceeds_extreme_priors():
    tox_prior = [0.1, 0.2]
    tox_target = 0.3
    tox_limit = 0.33
    eff_limit = 0.05
    skeletons = [[0.6, 0.4]]
    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)

    # Very narrow prior
    # norm(0, 1e-4) -> scale=1e-4
    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_limit,
        eff_limit,
        metric,
        first_dose=1,
        max_size=20,
        theta_prior=norm(loc=0, scale=1e-4),
    )

    p_quad = trial.prob_eff_exceeds(0.5, backend="quadrature")
    p_analytic = trial.prob_eff_exceeds(0.5, backend="analytic")

    # Should both be close to 1.0 or 0.0 depending on where the prior is centered
    # log(0.5)/log(0.6) = 1.35. Prior is at 0. So theta < 1.35 is very likely.
    assert np.allclose(p_quad, [1.0, 1.0], atol=1e-3)
    assert np.allclose(p_analytic, [1.0, 1.0], atol=1e-3)

    # Very wide prior
    trial = WATU(
        skeletons,
        tox_prior,
        tox_target,
        tox_limit,
        eff_limit,
        metric,
        first_dose=1,
        max_size=20,
        theta_prior=norm(0, 100),
    )
    # Give some data to pin it down
    trial.update([(1, 0, 1)] * 10 + [(2, 0, 0)] * 10)

    p_quad = trial.prob_eff_exceeds(0.5, backend="quadrature")
    p_mc = trial.prob_eff_exceeds(0.5, backend="mc", n=10**6)
    assert np.allclose(p_quad, p_mc, atol=0.01)
