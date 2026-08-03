
import numpy as np
import pytest

from clintrials.core.stats import (
    ProbabilityDensitySample,
)


@pytest.fixture
def prob_density_sample():
    np.random.seed(0)
    samp = np.random.rand(100, 2)
    func = lambda x: np.exp(-np.sum((x - 0.5) ** 2, axis=1))
    return ProbabilityDensitySample(samp, func)

def test_prob_density_sample_expectation(prob_density_sample):
    vector = prob_density_sample._samp[:, 0]
    exp = prob_density_sample.expectation(vector)
    assert np.isclose(exp, 0.5159530273133676)

def test_prob_density_sample_variance(prob_density_sample):
    vector = prob_density_sample._samp[:, 0]
    var = prob_density_sample.variance(vector)
    assert np.isclose(var, 0.07429207203225896)


