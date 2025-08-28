__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

""" Tests of the clintrials.dosefindings.crm module. """

import numpy as np
import pandas as pd
from scipy.stats import norm

from clintrials.core.math import (
    empiric,
    inverse_empiric,
    inverse_logistic,
    inverse_logit1,
    logistic,
    logit1,
)
from clintrials.dosefinding.crm import CRM


def setup_func():
    pass


def teardown_func():
    pass


def test_CRM_bayes():

    # Test that Bayesian CRM works by reproducing Table 3.2 on p.26 of Cheung's book:
    # Dose Finding By The Continual Reassessment Method, (Chapman & Hall/CRC Biostatistics Series)
    # Table 3.2 gives a patient-by-patient simulation of a CRM dose-finding trial, including
    # an estimate of the beta parameter at each iteration. If we can reproduce the doses delivered,
    # the presense or absence of toxic events, and the estimate of beta at each turn, that suggests
    # we are doing something right.

    # Trial simulation parameters. These are copied straight out of the book.
    true_toxicity = [0.02, 0.04, 0.10, 0.25, 0.50]
    doses = [3, 5, 5, 3, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    toxicity_events = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    tolerances = [
        0.571,
        0.642,
        0.466,
        0.870,
        0.634,
        0.390,
        0.524,
        0.773,
        0.175,
        0.627,
        0.321,
        0.099,
        0.383,
        0.995,
        0.628,
        0.346,
        0.919,
        0.022,
        0.647,
        0.469,
    ]
    beta_hats = [
        0.60,
        0.93,
        0.04,
        0.18,
        0.28,
        0.34,
        0.41,
        0.47,
        0.31,
        0.35,
        0.25,
        0.15,
        0.18,
        0.21,
        0.24,
        0.26,
        0.28,
        0.21,
        0.22,
        0.24,
    ]
    beta_hat_epsilon = 0.005

    # CRM parameters. These are required to make the CRM algo reproduce all of the above.
    prior = [0.05, 0.12, 0.25, 0.40, 0.55]
    toxicity_target = 0.25
    first_dose = 3
    F_func = logistic
    inverse_F = inverse_logistic
    beta_prior = norm(loc=0, scale=np.sqrt(1.34))

    # Our trial object
    crm = CRM(
        prior,
        toxicity_target,
        first_dose,
        max_size=len(tolerances),
        F_func=F_func,
        inverse_F=inverse_F,
        beta_prior=beta_prior,
        method="bayes",
        use_quick_integration=False,
        estimate_var=True,
    )
    dose = first_dose

    # Confirm that dose (x_i), toxicity (y_i), and beta estimate (hat(beta)_i) in Ken Cheung's
    # table can all be reproduced for each of the 20 patients:
    for patient_no in range(1, 21):
        assert dose == doses[patient_no - 1]
        toxicity = 1 if tolerances[patient_no - 1] < true_toxicity[dose - 1] else 0
        assert toxicity == toxicity_events[patient_no - 1]
        dose = crm.update([(dose, toxicity)])
        assert abs(crm.beta_hat - beta_hats[patient_no - 1]) <= beta_hat_epsilon


def test_CRM_mle():

    # Test that MLE CRM works by reproducing an example in Python that can be verified in R
    # using Cheung's dfcrm package.

    # Trial simulation parameters.
    doses = [3, 3, 1, 2, 2, 3, 3, 2, 3, 2, 1, 2, 1, 1, 1, 2, 2]
    toxicity_events = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    beta_hats = [
        np.nan,
        np.nan,
        -0.312,
        -0.193,
        -0.099,
        -0.040,
        0.030,
        -0.121,
        -0.084,
        -0.177,
        -0.284,
        -0.256,
        -0.336,
        -0.308,
        -0.286,
        -0.266,
        -0.240,
    ]
    beta_hat_epsilon = 0.005

    # CRM parameters. These are required to make the CRM algo reproduce all of the above.
    prior = [0.05, 0.12, 0.25, 0.40, 0.55]
    toxicity_target = 0.25
    first_dose = 3
    F_func = logistic
    inverse_F = inverse_logistic
    beta_prior = norm(loc=0, scale=np.sqrt(1.34))

    # Our trial object
    crm = CRM(
        prior,
        toxicity_target,
        first_dose,
        max_size=len(doses),
        F_func=F_func,
        inverse_F=inverse_F,
        beta_prior=beta_prior,
        method="mle",
    )
    # MLE CRM needs at least one of each toxicity event to proceed sensibly.
    # Cases 1 and 2 provide that:
    dose = crm.update([(doses[0], toxicity_events[0]), (doses[1], toxicity_events[1])])

    # Confirm that dose (x_i), toxicity (y_i), and beta estimate (hat(beta)_i) in Ken Cheung's
    # table can all be reproduced for each of the 20 patients:
    for patient_no in range(2, len(doses)):
        assert dose == doses[patient_no]
        assert abs(crm.beta_hat - beta_hats[patient_no]) <= beta_hat_epsilon
        toxicity = toxicity_events[patient_no]
        dose = crm.update([(dose, toxicity)])

    # This is all verifiable in R.


def test_CRM_bayes_again():
    prior = [0.1, 0.2, 0.4, 0.6]
    target = 0.4
    doses = [1, 1, 1, 2, 2, 2]
    tox = [0, 0, 0, 1, 0, 1]
    cases = list(zip(doses, tox))
    trial_plugin_1 = CRM(
        prior,
        target,
        1,
        30,
        F_func=empiric,
        inverse_F=inverse_empiric,
        use_quick_integration=False,
        plugin_mean=True,
    )
    trial_plugin_2 = CRM(
        prior,
        target,
        1,
        30,
        F_func=empiric,
        inverse_F=inverse_empiric,
        use_quick_integration=True,
        plugin_mean=True,
    )
    trial_plugin_3 = CRM(
        prior,
        target,
        1,
        30,
        F_func=logistic,
        inverse_F=inverse_logistic,
        use_quick_integration=False,
        plugin_mean=True,
    )
    trial_plugin_4 = CRM(
        prior,
        target,
        1,
        30,
        F_func=logistic,
        inverse_F=inverse_logistic,
        use_quick_integration=True,
        plugin_mean=True,
    )
    trial_plugin_1.update(cases)
    trial_plugin_2.update(cases)
    trial_plugin_3.update(cases)
    trial_plugin_4.update(cases)

    assert np.all(
        np.array(trial_plugin_1.prob_tox()) - np.array([[0.240, 0.368, 0.566, 0.728]])
        < 0.001
    )
    assert np.all(
        np.array(trial_plugin_2.prob_tox()) - np.array([[0.240, 0.368, 0.566, 0.728]])
        < 0.001
    )
    assert np.all(
        np.array(trial_plugin_3.prob_tox()) - np.array([[0.274, 0.412, 0.598, 0.734]])
        < 0.001
    )
    assert np.all(
        np.array(trial_plugin_4.prob_tox()) - np.array([[0.274, 0.412, 0.598, 0.734]])
        < 0.001
    )
    # These are verifiable in R


class TestCRMMLEVariance:
    def setup_method(self):
        self.prior = [0.05, 0.12, 0.25, 0.40, 0.55]
        self.target = 0.25
        self.F_func = logistic
        self.inverse_F = inverse_logistic
        self.beta_prior = norm(loc=0, scale=np.sqrt(1.34))
        self.doses = [3, 3, 1, 2, 2, 3, 3, 2, 3, 2, 1, 2, 1, 1, 1, 2, 2]
        self.tox = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]

    def test_mle_variance_hessian(self):
        trial = CRM(
            self.prior,
            self.target,
            first_dose=3,
            max_size=len(self.doses),
            F_func=self.F_func,
            inverse_F=self.inverse_F,
            beta_prior=self.beta_prior,
            method="mle",
            estimate_var=True,
            mle_var_method="hessian",
        )
        trial.update(list(zip(self.doses, self.tox)))
        assert trial.beta_var is not None
        assert trial.beta_var > 0

    def test_mle_variance_bootstrap(self):
        trial = CRM(
            self.prior,
            self.target,
            first_dose=3,
            max_size=len(self.doses),
            F_func=self.F_func,
            inverse_F=self.inverse_F,
            beta_prior=self.beta_prior,
            method="mle",
            estimate_var=True,
            mle_var_method="bootstrap",
            bootstrap_samples=50,  # Smaller sample for faster test
        )
        trial.update(list(zip(self.doses, self.tox)))
        assert trial.beta_var is not None
        assert trial.beta_var > 0

    def test_mle_variance_comparison(self):
        trial_hessian = CRM(
            self.prior,
            self.target,
            first_dose=3,
            max_size=len(self.doses),
            F_func=self.F_func,
            inverse_F=self.inverse_F,
            beta_prior=self.beta_prior,
            method="mle",
            estimate_var=True,
            mle_var_method="hessian",
        )
        trial_hessian.update(list(zip(self.doses, self.tox)))

        trial_bootstrap = CRM(
            self.prior,
            self.target,
            first_dose=3,
            max_size=len(self.doses),
            F_func=self.F_func,
            inverse_F=self.inverse_F,
            beta_prior=self.beta_prior,
            method="mle",
            estimate_var=True,
            mle_var_method="bootstrap",
            bootstrap_samples=200,
        )
        trial_bootstrap.update(list(zip(self.doses, self.tox)))

        assert np.isclose(trial_hessian.beta_var, trial_bootstrap.beta_var, rtol=0.3)


def test_CRM_class_with_bcrm_fixtures():
    # Load the fixtures
    expected_probs = pd.read_csv("tests/fixtures/expected_posterior_dlt_probs.csv")
    expected_doses = pd.read_csv("tests/fixtures/next_dose_recommendations.csv")

    # Scenario 1
    p_tox_prior_1 = [0.1, 0.2, 0.3, 0.4]
    target_tox_1 = 0.3
    doses_1 = [1, 1, 2, 2, 3, 3]
    dlt_1 = [0, 0, 0, 1, 1, 1]
    cases_1 = list(zip(doses_1, dlt_1))

    trial1 = CRM(
        prior=p_tox_prior_1,
        target=target_tox_1,
        first_dose=1,
        max_size=30,
        F_func=logit1,
        inverse_F=inverse_logit1,
        beta_prior=norm(loc=0, scale=np.sqrt(1.34)),
        method="bayes",
        plugin_mean=False,
    )
    trial1.update(cases_1)
    prob_tox_1 = trial1.prob_tox()
    recommended_dose_1 = trial1.next_dose()

    expected_prob_tox_1 = expected_probs[expected_probs.scenario == 1].prob.values
    expected_next_dose_1 = expected_doses[
        expected_doses.scenario == 1
    ].next_dose.values[0]

    assert np.allclose(prob_tox_1, expected_prob_tox_1, atol=1e-2)
    assert recommended_dose_1 == expected_next_dose_1

    # Scenario 2
    p_tox_prior_2 = [0.05, 0.1, 0.2, 0.35, 0.5]
    target_tox_2 = 0.2
    doses_2 = [1, 1, 1, 2, 2, 2]
    dlt_2 = [0, 0, 0, 0, 0, 0]
    cases_2 = list(zip(doses_2, dlt_2))

    trial2 = CRM(
        prior=p_tox_prior_2,
        target=target_tox_2,
        first_dose=1,
        max_size=30,
        F_func=logit1,
        inverse_F=inverse_logit1,
        beta_prior=norm(loc=0, scale=np.sqrt(1.34)),
        method="bayes",
        plugin_mean=False,
    )
    trial2.update(cases_2)
    prob_tox_2 = trial2.prob_tox()
    recommended_dose_2 = trial2.next_dose()

    expected_prob_tox_2 = expected_probs[expected_probs.scenario == 2].prob.values
    expected_next_dose_2 = expected_doses[
        expected_doses.scenario == 2
    ].next_dose.values[0]

    assert np.allclose(prob_tox_2, expected_prob_tox_2, atol=1e-1)
    assert recommended_dose_2 == expected_next_dose_2
