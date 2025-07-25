__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Tests of the clintrials.dosefindings.wagestait module. """

from nose.tools import with_setup
import numpy as np
from scipy.stats import norm

from clintrials.common import empiric, logistic, inverse_empiric, inverse_logistic
from clintrials.dosefinding.wagestait import WagesTait


def setup_func():
    pass


def teardown_func():
    pass


@with_setup(setup_func, teardown_func)
def test_wages_tait_1():

    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30

    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        [0.30, 0.40, 0.50, 0.60, 0.50, 0.40],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.60, 0.60],
        [0.40, 0.50, 0.60, 0.60, 0.60, 0.60],
        [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    ]

    first_dose = 1
    trial_size = 64
    ar_size = 16

    trial = WagesTait(skeletons, tox_prior, tox_target, tox_cutoff, eff_cutoff, first_dose, trial_size, ar_size)

    cases = [
        (1,1,0), (1,0,0), (1,0,0),
        (2,0,0), (2,0,0), (2,0,1),
        (3,1,1), (3,0,1),
    ]

    trial.update(cases)
    # No idea what this will be because it is randomised

    assert np.all(np.abs(trial.post_tox_probs - np.array([
        0.1374908, 0.3126617, 0.4095831, 0.4856057, 0.5506505, 0.6086650
    ])) < 0.001)  # The first one varies a bit more
    assert np.all(np.abs(trial.post_eff_probs - np.array([
        0.2479070, 0.3639813, 0.4615474, 0.5497718, 0.6321674, 0.7105235
    ])) < 0.00001)
    assert np.all(np.abs(trial.w - np.array([
        0.01347890, 0.03951504, 0.12006585, 0.11798287, 0.11764227, 0.12346595,
        0.11764227, 0.11798287, 0.12006585, 0.07073296, 0.04142517
    ])) < 0.00001)
    assert trial.most_likely_model_index == 5
    assert trial.admissable_set() == [1, 2]
    assert np.abs(trial.dose_toxicity_lower_bound(1) - 0.008403759) < 0.00001

    # The exact values above were taken from Nolan's implementation in R.



def test_wages_tait_2():

    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30

    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        [0.30, 0.40, 0.50, 0.60, 0.50, 0.40],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.60, 0.60],
        [0.40, 0.50, 0.60, 0.60, 0.60, 0.60],
        [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    ]

    first_dose = 1
    trial_size = 64
    ar_size = 16

    trial = WagesTait(skeletons, tox_prior, tox_target, tox_cutoff, eff_cutoff, first_dose, trial_size, ar_size)

    cases = [
        (1,1,0), (1,0,0), (1,0,0),
        (2,0,0), (2,0,0), (2,0,1),
        (3,1,1), (3,0,1), (3,1,1),
        (2,0,0), (2,0,0), (2,1,1),
        (3,0,1), (3,0,0), (3,1,1),
        (4,1,1), (4,0,1), (4,0,1),
    ]

    next_dose = trial.update(cases)
    assert next_dose == 2
    assert np.all(trial.post_tox_probs - np.array([
        0.1292270, 0.3118714, 0.4124383, 0.4906021, 0.5569093, 0.6155878
    ]) < 0.00001)
    assert np.all(trial.post_eff_probs - np.array([
        0.3999842, 0.4935574, 0.5830685, 0.6697646, 0.5830685, 0.4935574
    ]) < 0.00001)
    assert np.all(trial.w - np.array([
        0.00165319, 0.00650976, 0.06932715, 0.15695883, 0.14129752, 0.14465125,
        0.14129752, 0.15695883, 0.11767193, 0.04176601, 0.02190798
    ]) < 0.00001)
    assert trial.most_likely_model_index == 3
    assert trial.admissable_set() == [1, 2]
    assert trial.dose_toxicity_lower_bound(1) - 0.008403759 < 0.00001
    assert trial.dose_efficacy_upper_bound(next_dose) - 0.7772219 < 0.00001

    # The exact values above were taken from Nolan's implementation in R.
