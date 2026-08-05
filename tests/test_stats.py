# SPDX-License-Identifier: MIT


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
    vector = prob_density_sample.samples[:, 0]
    exp = prob_density_sample.expectation(vector)
    assert np.isclose(exp, 0.5159530273133676)

def test_prob_density_sample_variance(prob_density_sample):
    vector = prob_density_sample.samples[:, 0]
    var = prob_density_sample.variance(vector)
    assert np.isclose(var, 0.07429207203225896)

def test_prob_density_sample_samples_property(prob_density_sample):
    assert isinstance(prob_density_sample.samples, np.ndarray)
    assert prob_density_sample.samples.shape == (100, 2)
    # Check read-only
    with pytest.raises(AttributeError):
        prob_density_sample.samples = np.random.rand(100, 2)

def test_prob_density_sample_resample(prob_density_sample):
    boot_samps = 1000
    resampled = prob_density_sample.resample(boot_samps, rng=np.random.default_rng(42))
    assert resampled.shape == (boot_samps, 2)
    # All resampled elements must be from the original samples
    for item in resampled:
        assert any(np.allclose(item, original) for original in prob_density_sample.samples)


