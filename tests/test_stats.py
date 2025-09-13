import numpy as np
import pytest
from collections import OrderedDict
from clintrials.core.stats import or_test, chi_squ_test, bootstrap, beta_like_normal, ProbabilityDensitySample

def test_or_test_simple_case():
    """
    Test or_test with a simple case.
    """
    result = or_test(a=20, b=5, c=10, d=15)
    expected_or = (20 * 15) / (10 * 5)
    expected_log_or_se = np.sqrt(1/20 + 1/5 + 1/10 + 1/15)

    assert result['OR'] == expected_or
    assert np.isclose(result['Log(OR) SE'], expected_log_or_se)

    # Calculate expected CI
    from scipy.stats import norm
    ci_alpha = 0.05
    ci_scalars = norm.ppf([ci_alpha / 2, 1 - ci_alpha / 2])
    or_ci = np.exp(np.log(expected_or) + ci_scalars * expected_log_or_se)
    assert np.allclose(result['OR CI'], or_ci)
    assert result['Alpha'] == ci_alpha


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
    chi_sq_stat = ((obs - expected)**2 / expected).sum()

    assert np.isclose(result['TestStatistic'], chi_sq_stat)
    assert result['Df'] == 1

    odds = result['Odds']
    a, b, c, d = 2, 3, 3, 2
    assert odds['ABCD'] == [a, b, c, d]
    assert np.isclose(odds['OR'], (a*d)/(b*c))

    log_or_se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    assert np.isclose(odds['Log(OR) SE'], log_or_se)

    from scipy.stats import norm
    ci_alpha = 0.05
    ci_scalars = norm.ppf([ci_alpha / 2, 1 - ci_alpha / 2])
    or_ci = np.exp(np.log((a*d)/(b*c)) + ci_scalars * log_or_se)
    assert np.allclose(odds['OR CI'], or_ci)
    assert odds['Alpha'] == ci_alpha

def test_bootstrap():
    """
    Test bootstrap function.
    """
    x = [1, 2, 3, 4, 5]
    boot_sample = bootstrap(x)
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
    func = lambda x: np.exp(-np.sum((x - 0.5)**2, axis=1))
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
    assert np.isclose(q, 0.5562926340774739)

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
    assert np.isclose(q, 0.5562926340774739)
