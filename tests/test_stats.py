from collections import OrderedDict

import numpy as np
import pytest

from clintrials.core.stats import (
    ProbabilityDensitySample,
    beta_like_normal,
    bootstrap,
    chi_squ_test,
    correlation_ci,
    or_test,
)


def test_or_test_simple_case():
    """
    Test or_test with a simple case.
    """
    result = or_test(a=20, b=5, c=10, d=15)
    expected_or = (20 * 15) / (10 * 5)
    expected_log_or_se = np.sqrt(1 / 20 + 1 / 5 + 1 / 10 + 1 / 15)

    assert result["OR"] == expected_or
    assert np.isclose(result["Log(OR) SE"], expected_log_or_se)

    # Calculate expected CI
    from scipy.stats import norm

    ci_alpha = 0.05
    ci_scalars = norm.ppf([ci_alpha / 2, 1 - ci_alpha / 2])
    or_ci = np.exp(np.log(expected_or) + ci_scalars * expected_log_or_se)
    assert np.allclose(result["OR CI"], or_ci)
    assert result["Alpha"] == ci_alpha


def test_chi_squ_test_2x2_case():
    """
    Test chi_squ_test with a 2x2 table.
    """
    x = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    result = chi_squ_test(x, y)

    # Manually calculate chi-squared statistic
    obs = np.array([[2, 3], [3, 2]])
    row_totals = obs.sum(axis=1)
    col_totals = obs.sum(axis=0)
    grand_total = obs.sum()
    expected = np.outer(row_totals, col_totals) / grand_total
    chi_sq_stat = ((obs - expected) ** 2 / expected).sum()

    assert np.isclose(result["TestStatistic"], chi_sq_stat)
    assert result["Df"] == 1

    odds = result["Odds"]
    a, b, c, d = 2, 3, 3, 2
    assert odds["ABCD"] == [a, b, c, d]
    assert np.isclose(odds["OR"], (a * d) / (b * c))

    log_or_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    assert np.isclose(odds["Log(OR) SE"], log_or_se)

    from scipy.stats import norm

    ci_alpha = 0.05
    ci_scalars = norm.ppf([ci_alpha / 2, 1 - ci_alpha / 2])
    or_ci = np.exp(np.log((a * d) / (b * c)) + ci_scalars * log_or_se)
    assert np.allclose(odds["OR CI"], or_ci)
    assert odds["Alpha"] == ci_alpha


def test_bootstrap():
    """
    Test bootstrap function.
    """
    import numpy as np
    rng = np.random.default_rng(123)
    x = [1, 2, 3, 4, 5]
    boot_sample = bootstrap(x, rng)
    assert len(boot_sample) == len(x)
    assert all(item in x for item in boot_sample)


def test_beta_like_normal():
    """
    Test beta_like_normal function.
    """
    mu, sigma = 0.5, 0.1
    alpha, beta = beta_like_normal(mu, sigma)
    assert np.isclose(alpha, 12)
    assert np.isclose(beta, 12)


@pytest.fixture
def prob_density_sample():
    """
    Fixture for ProbabilityDensitySample.
    """
    np.random.seed(0)
    samp = np.random.rand(100, 2)
    func = lambda x: np.exp(-np.sum((x - 0.5) ** 2, axis=1))
    return ProbabilityDensitySample(samp, func)


def test_prob_density_sample_expectation(prob_density_sample):
    """
    Test ProbabilityDensitySample.expectation.
    """
    vector = prob_density_sample._samp[:, 0]
    exp = prob_density_sample.expectation(vector)
    assert np.isclose(exp, 0.5159530273133676)


def test_prob_density_sample_variance(prob_density_sample):
    """
    Test ProbabilityDensitySample.variance.
    """
    vector = prob_density_sample._samp[:, 0]
    var = prob_density_sample.variance(vector)
    assert np.isclose(var, 0.07429207203225896)


def test_prob_density_sample_cdf(prob_density_sample):
    """
    Test ProbabilityDensitySample.cdf.
    """
    cdf_val = prob_density_sample.cdf(0, 0.5)
    assert np.isclose(cdf_val, 0.4336043884413059)


def test_prob_density_sample_quantile(prob_density_sample):
    """
    Test ProbabilityDensitySample.quantile.
    """
    q = prob_density_sample.quantile(0, 0.5)
    assert np.isclose(q, 0.566836066294248)


def test_prob_density_sample_cdf_vector(prob_density_sample):
    """
    Test ProbabilityDensitySample.cdf_vector.
    """
    vector = prob_density_sample._samp[:, 0]
    cdf_val = prob_density_sample.cdf_vector(vector, 0.5)
    assert np.isclose(cdf_val, 0.4336043884413059)


def test_prob_density_sample_quantile_vector(prob_density_sample):
    """
    Test ProbabilityDensitySample.quantile_vector.
    """
    vector = prob_density_sample._samp[:, 0]
    q = prob_density_sample.quantile_vector(vector, 0.5)
    assert np.isclose(q, 0.566836066294248)


def test_correlation_ci_fisher():
    """
    Test correlation_ci with method='fisher'.
    """
    # Example from a textbook: r = 0.7, n = 50, alpha = 0.05
    # z = arctanh(0.7) = 0.8673
    # se = 1 / sqrt(50 - 3) = 0.14586
    # z_crit = 1.96
    # z_ci = [0.8673 - 1.96 * 0.14586, 0.8673 + 1.96 * 0.14586]
    # z_ci = [0.5814, 1.1532]
    # r_ci = [tanh(0.5814), tanh(1.1532)] = [0.5236, 0.8190]

    r = 0.7
    n = 50
    result = correlation_ci(r=r, n=n, method="fisher")

    expected_low = np.tanh(np.arctanh(r) - 1.959963984540054 * (1 / np.sqrt(n - 3)))
    expected_high = np.tanh(np.arctanh(r) + 1.959963984540054 * (1 / np.sqrt(n - 3)))

    assert np.isclose(result[0], expected_low)
    assert np.isclose(result[1], r)
    assert np.isclose(result[2], expected_high)

    # Edge case r=1
    result_1 = correlation_ci(r=1.0, n=50, method="fisher")
    assert np.all(result_1 == 1.0)

    # Edge case r=-1
    result_neg_1 = correlation_ci(r=-1.0, n=50, method="fisher")
    assert np.all(result_neg_1 == -1.0)

    # Validation
    with pytest.raises(ValueError):
        correlation_ci(r=0.5, n=3, method="fisher")
    with pytest.raises(ValueError):
        correlation_ci(r=1.1, n=50, method="fisher")


def test_correlation_ci_bayes():
    """
    Test correlation_ci with method='bayes'.
    """
    samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = correlation_ci(samples=samples, alpha=0.1, method="bayes")

    # alpha/2 = 0.05, 1 - alpha/2 = 0.95
    # np.quantile defaults to linear interpolation
    expected_low = np.quantile(samples, 0.05)
    expected_high = np.quantile(samples, 0.95)
    expected_mean = np.mean(samples)

    assert np.isclose(result[0], expected_low)
    assert np.isclose(result[1], expected_mean)
    assert np.isclose(result[2], expected_high)


def test_correlation_ci_bayes_weighted():
    """
    Test correlation_ci with method='bayes' and weights.
    """
    samples = np.array([0.1, 0.9])
    weights = np.array([0.9, 0.1])
    # Weighted mean: 0.1 * 0.9 + 0.9 * 0.1 = 0.09 + 0.09 = 0.18
    # Cumulative weights: [0.9, 1.0]
    # alpha=0.5 => alpha/2 = 0.25, 1 - alpha/2 = 0.75
    # 0.25 is before 0.9, so low = 0.1 (interp will handle it)
    # 0.75 is before 0.9, so high = 0.1

    result = correlation_ci(samples=samples, weights=weights, alpha=0.5, method="bayes")

    assert np.isclose(result[1], 0.18)
    assert np.isclose(result[0], 0.1)
    assert np.isclose(result[2], 0.1)

    # More complex weighted case
    samples = np.array([0.1, 0.5, 0.9])
    weights = np.array([1, 1, 1])
    result = correlation_ci(
        samples=samples, weights=weights, alpha=0.05, method="bayes"
    )
    unweighted = correlation_ci(samples=samples, alpha=0.05, method="bayes")
    # np.quantile and np.interp might differ slightly in how they handle discrete values
    assert np.isclose(result[1], unweighted[1])
