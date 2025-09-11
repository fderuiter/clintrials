__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

"""
This module contains the first version of classes and functions for the PePS2
trial. PePS2 studies the efficacy and toxicity of a drug in a population of
performance status 2 lung cancer patients. Patient outcomes may be affected by
whether they have been treated before and their PD-L1 expression rate.

The trial uses the BeBOP design to incorporate this predictive data to find
the subpopulation(s) where the drug is effective and tolerable.
"""

import datetime
import glob
import json
import logging
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd

from clintrials.core.stats import ProbabilityDensitySample, chi_squ_test, or_test
from clintrials.utils import (
    atomic_to_json,
    correlated_binary_outcomes,
    iterable_to_json,
)

logger = logging.getLogger(__name__)


def pi_t(disease_status, mutation_status, alpha0=0, alpha1=0, alpha2=0):
    """Calculates the probability of toxicity.

    Args:
        disease_status (int): 1 if pre-treated, 0 otherwise.
        mutation_status (int): 1 if biomarker positive, 0 otherwise.
        alpha0 (float, optional): Intercept parameter. Defaults to 0.
        alpha1 (float, optional): Disease status coefficient. Defaults to 0.
        alpha2 (float, optional): Mutation status coefficient. Defaults to 0.

    Returns:
        float: The probability of toxicity.
    """
    z = alpha0 + alpha1 * disease_status + alpha2 * mutation_status
    response = 1 / (1 + np.exp(-z))
    return response


def pi_e(disease_status, mutation_status, beta0=0, beta1=0, beta2=0):
    """Calculates the probability of efficacy.

    Args:
        disease_status (int): 1 if pre-treated, 0 otherwise.
        mutation_status (int): 1 if biomarker positive, 0 otherwise.
        beta0 (float, optional): Intercept parameter. Defaults to 0.
        beta1 (float, optional): Disease status coefficient. Defaults to 0.
        beta2 (float, optional): Mutation status coefficient. Defaults to 0.

    Returns:
        float: The probability of efficacy.
    """
    z = beta0 + beta1 * disease_status + beta2 * mutation_status
    return 1 / (1 + np.exp(-z))


def pi_ab(
    disease_status, mutation_status, eff, tox, alpha0, beta0, beta1, beta2, psi
):
    """Calculates the likelihood of a joint efficacy-toxicity outcome.

    Args:
        disease_status (int): 1 if pre-treated, 0 otherwise.
        mutation_status (int): 1 if biomarker positive, 0 otherwise.
        eff (int): Efficacy outcome (1 or 0).
        tox (int): Toxicity outcome (1 or 0).
        alpha0 (float): Toxicity intercept.
        beta0 (float): Efficacy intercept.
        beta1 (float): Efficacy disease status coefficient.
        beta2 (float): Efficacy mutation status coefficient.
        psi (float): Correlation parameter.

    Returns:
        float: The likelihood of the outcome.
    """
    a = tox
    b = eff
    p1 = pi_t(disease_status, mutation_status, alpha0)
    p2 = pi_e(disease_status, mutation_status, beta0, beta1, beta2)
    response = p1**a * (1 - p1) ** (1 - a) * p2**b * (1 - p2) ** (1 - b)
    response = response + (-1) ** (a + b) * p1 * (1 - p1) * p2 * (1 - p2) * (
        np.exp(psi) - 1
    ) / (np.exp(psi) + 1)
    return response


def l_n(cases, alpha0, beta0, beta1, beta2, psi):
    """Calculates the compound likelihood for a set of cases.

    Args:
        cases (list): A list of patient cases.
        alpha0 (float): Toxicity intercept.
        beta0 (float): Efficacy intercept.
        beta1 (float): Efficacy disease status coefficient.
        beta2 (float): Efficacy mutation status coefficient.
        psi (float): Correlation parameter.

    Returns:
        float: The compound likelihood.
    """
    if len(cases) > 0:
        disease_status, mutation_status, eff, tox = zip(*cases)
        response = np.ones_like(alpha0)
        for i in range(len(tox)):
            p = pi_ab(
                disease_status[i],
                mutation_status[i],
                eff[i],
                tox[i],
                alpha0,
                beta0,
                beta1,
                beta2,
                psi,
            )
            response = response * p
        return response
    else:
        return np.ones_like(alpha0)


def get_posterior_probs(
    cases, priors, tox_cutoffs, eff_cutoffs, n=10**5, epsilon=1e-5
):
    """Gets the posterior probabilities for the PePS2 trial using the BeBOP design.

    This function calculates the posterior probabilities of efficacy and toxicity for each of the
    four subgroups in the PePS2 trial, which are defined by two binary covariates:
    pre-treatment status and PD-L1 expression.

    The calculation is done using Monte Carlo integration.

    Args:
        cases (list): A list of 4-tuples, where each tuple represents a patient and contains:
            (disease_status, mutation_status, efficacy, toxicity).
            - disease_status (int): 1 if patient has been pre-treated, 0 otherwise.
            - mutation_status (int): 1 if patient is PD-L1 positive, 0 otherwise.
            - efficacy (int): 1 for an efficacious outcome, 0 for alternative.
            - toxicity (int): 1 for a toxic event, 0 for tolerance event.
        priors (list): A list of prior distributions for the model parameters:
            - alpha_0 (scipy.stats.rv_continuous): Prior for the toxicity intercept.
            - beta_0 (scipy.stats.rv_continuous): Prior for the efficacy intercept.
            - beta_1 (scipy.stats.rv_continuous): Prior for the efficacy effect of pre-treatment.
            - beta_2 (scipy.stats.rv_continuous): Prior for the efficacy effect of PD-L1 status.
            - psi (scipy.stats.rv_continuous): Prior for the correlation parameter.
        tox_cutoffs (list or float): The desired maximum toxicity cutoffs for the four groups.
        eff_cutoffs (list or float): The desired minimum efficacy cutoffs for the four groups.
        n (int, optional): Number of random points to use in Monte Carlo integration. Defaults to 10**5.
        epsilon (float, optional): A small number to define the integration range via the ppf of the priors. Defaults to 1e-5.

    Returns:
        tuple: A tuple containing:
            - A nested list of posterior probabilities: [Prob(Toxicity), Prob(Efficacy),
              Prob(Toxicity < cutoff), Prob(Efficacy > cutoff)], for each patient cohort.
              The cohorts are ordered: (Not pre-treated, PD-L1 neg), (Not pre-treated, PD-L1 pos),
              (Pre-treated, PD-L1 neg), (Pre-treated, PD-L1 pos).
            - The ProbabilityDensitySample object used for the calculations.
    """
    if not isinstance(tox_cutoffs, list):
        tox_cutoffs = [tox_cutoffs] * 4
    if not isinstance(eff_cutoffs, list):
        eff_cutoffs = [eff_cutoffs] * 4

    limits = [(dist.ppf(epsilon), dist.ppf(1 - epsilon)) for dist in priors]
    samp = np.column_stack(
        [np.random.uniform(*limit_pair, size=n) for limit_pair in limits]
    )

    lik_integrand = (
        lambda x: l_n(cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])
        * priors[0].pdf(x[:, 0])
        * priors[1].pdf(x[:, 1])
        * priors[2].pdf(x[:, 2])
        * priors[3].pdf(x[:, 3])
        * priors[4].pdf(x[:, 4])
    )
    pds = ProbabilityDensitySample(samp, lik_integrand)

    probs = []
    patient_cohorts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (disease_status, mutation_status), tox_cutoff, eff_cutoff in zip(
        patient_cohorts, tox_cutoffs, eff_cutoffs
    ):

        tox_probs = pi_t(disease_status, mutation_status, alpha0=samp[:, 0])
        eff_probs = pi_e(
            disease_status,
            mutation_status,
            beta0=samp[:, 1],
            beta1=samp[:, 2],
            beta2=samp[:, 3],
        )

        probs.append(
            (
                pds.expectation(tox_probs),
                pds.expectation(eff_probs),
                pds.expectation(tox_probs < tox_cutoff),
                pds.expectation(eff_probs > eff_cutoff),
            )
        )

    return probs, pds


class PePS2BeBOP:
    """A class to manage a PePS2 trial using the BeBOP design."""

    def __init__(
        self,
        theta_priors,
        tox_cutoffs,
        eff_cutoffs,
        tox_certainty,
        eff_certainty,
        epsilon=1e-5,
    ):
        """Initializes a PePS2BeBOP trial object.

        Args:
            theta_priors (list): A list of prior distributions for the model
                parameters.
            tox_cutoffs (list or float): The maximum acceptable toxicity
                probabilities.
            eff_cutoffs (list or float): The minimum acceptable efficacy
                probabilities.
            tox_certainty (list or float): The posterior certainty required for
                acceptable toxicity.
            eff_certainty (list or float): The posterior certainty required for
                acceptable efficacy.
            epsilon (float, optional): A small number to define the integration
                range. Defaults to 1e-5.
        """
        self.priors = theta_priors
        self.tox_cutoffs = tox_cutoffs
        self.eff_cutoffs = eff_cutoffs
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty
        self.epsilon = epsilon

        self.reset()

    def reset(self):
        """Resets the trial to its initial state."""
        self.cases = []
        self.prob_tox = []
        self.prob_eff = []
        self.prob_acc_tox = []
        self.prob_acc_eff = []

    def size(self):
        """Gets the current number of patients in the trial.

        Returns:
            int: The number of patients.
        """
        return len(self.cases)

    def pretreated_statuses(self):
        """Gets the pre-treated statuses of all patients.

        Returns:
            list[int]: A list of pre-treated statuses.
        """
        return [case[0] for case in self.cases]

    def mutation_statuses(self):
        """Gets the mutation statuses of all patients.

        Returns:
            list[int]: A list of mutation statuses.
        """
        return [case[1] for case in self.cases]

    def efficacies(self):
        """Gets the efficacy outcomes of all patients.

        Returns:
            list[int]: A list of efficacy outcomes.
        """
        return [case[2] for case in self.cases]

    def toxicities(self):
        """Gets the toxicity outcomes of all patients.

        Returns:
            list[int]: A list of toxicity outcomes.
        """
        return [case[3] for case in self.cases]

    def update(self, cases, n=10**6, **kwargs):
        """Updates the trial with new patient cases.

        Args:
            cases (list): A list of new patient cases.
            n (int, optional): The number of samples for Monte Carlo
                integration. Defaults to 10**6.
            **kwargs: Additional keyword arguments.

        Returns:
            int: Always returns 0.
        """
        for disease_status, mutation_status, eff, tox in cases:
            self.cases.append((disease_status, mutation_status, eff, tox))

        post_probs, pds = get_posterior_probs(
            self.cases,
            self.priors,
            self.tox_cutoffs,
            self.eff_cutoffs,
            n,
            self.epsilon,
        )
        self._update(post_probs)
        self._pds = pds
        return 0

    def decision(self, i):
        """Determines the trial outcome decision for a specific group.

        Args:
            i (int): The index of the group.

        Returns:
            bool: `True` if the treatment is approved for the group, `False`
                otherwise.
        """
        eff_prob_hurdle = (
            self.eff_certainty[i]
            if isinstance(self.eff_certainty, list)
            else self.eff_certainty
        )
        tox_prob_hurdle = (
            self.tox_certainty[i]
            if isinstance(self.tox_certainty, list)
            else self.tox_certainty
        )
        return (
            self.prob_acc_eff[i] >= eff_prob_hurdle
            and self.prob_acc_tox[i] >= tox_prob_hurdle
        )

    def efficacy_effect(self, j, alpha=0.05):
        """Gets the confidence interval and mean estimate for an efficacy effect.

        Args:
            j (int): The index of the effect (0 for intercept, 1 for
                pre-treated, 2 for mutation).
            alpha (float, optional): The significance level for the confidence
                interval. Defaults to 0.05.

        Returns:
            numpy.ndarray: An array containing the lower bound, mean, and
                upper bound of the odds ratio.
        """
        if j == 0:
            expected_log_or = self._pds.expectation(self._pds._samp[:, 1])
            return np.exp(
                [
                    self._pds.quantile(1, alpha / 2),
                    expected_log_or,
                    self._pds.quantile(1, 1 - alpha / 2),
                ]
            )
        elif j == 1:
            expected_log_or = self._pds.expectation(self._pds._samp[:, 2])
            return np.exp(
                [
                    self._pds.quantile(2, alpha / 2),
                    expected_log_or,
                    self._pds.quantile(2, 1 - alpha / 2),
                ]
            )
        elif j == 2:
            expected_log_or = self._pds.expectation(self._pds._samp[:, 3])
            return np.exp(
                [
                    self._pds.quantile(3, alpha / 2),
                    expected_log_or,
                    self._pds.quantile(3, 1 - alpha / 2),
                ]
            )
        else:
            return (0, 0, 0)

    def toxicity_effect(self, j=0, alpha=0.05):
        """Gets the confidence interval and mean estimate for a toxicity effect.

        Args:
            j (int, optional): The index of the effect (0 for intercept).
                Defaults to 0.
            alpha (float, optional): The significance level for the confidence
                interval. Defaults to 0.05.

        Returns:
            numpy.ndarray: An array containing the lower bound, mean, and
                upper bound of the odds ratio.
        """
        if j == 0:
            expected_log_or = self._pds.expectation(self._pds._samp[:, 0])
            return np.exp(
                [
                    self._pds.quantile(0, alpha / 2),
                    expected_log_or,
                    self._pds.quantile(0, 1 - alpha / 2),
                ]
            )
        else:
            return (0, 0, 0)

    def correlation_effect(self, alpha=0.05):
        """Gets the confidence interval and mean estimate for the correlation
        between efficacy and toxicity.

        Args:
            alpha (float, optional): The significance level for the confidence
                interval. Defaults to 0.05.

        Returns:
            numpy.ndarray: An array containing the lower bound, mean, and
                upper bound of the correlation.
        """
        expected_psi = self._pds.expectation(self._pds._samp[:, 4])
        psi_levels = np.array(
            [
                self._pds.quantile(4, alpha / 2),
                expected_psi,
                self._pds.quantile(4, 1 - alpha / 2),
            ]
        )
        return (np.exp(psi_levels) - 1) / (np.exp(psi_levels) + 1)

    def _update(self, post_probs):
        prob_tox, prob_eff, prob_acc_tox, prob_acc_eff = zip(*post_probs)
        self.prob_tox = prob_tox
        self.prob_eff = prob_eff
        self.prob_acc_tox = prob_acc_tox
        self.prob_acc_eff = prob_acc_eff
        return 0


def simulate_peps2_patients(num_patients, prob_pretreated, prob_mutated, params):
    """Simulates a cohort of patients for the PePS2 trial.

    Args:
        num_patients (int): The number of patients to simulate.
        prob_pretreated (float): The probability of a patient being
            pre-treated.
        prob_mutated (float): The probability of a patient being biomarker
            positive.
        params (list[tuple[float, float, float]]): A list of tuples, where
            each tuple contains the probability of efficacy, probability of
            toxicity, and the efficacy-toxicity odds ratio for a subgroup.

    Returns:
        tuple: A tuple containing:
            - A numpy array with patient data (PreTreated, PD-L1+, Efficacy,
              Toxicity).
            - A tuple of group sizes (C00, C01, C10, C11).
    """
    subsample_sizes = np.random.multinomial(
        num_patients,
        [
            (1 - prob_pretreated) * (1 - prob_mutated),
            (1 - prob_pretreated) * prob_mutated,
            prob_pretreated * (1 - prob_mutated),
            prob_pretreated * prob_mutated,
        ],
        size=1,
    )[0]
    n00, n01, n10, n11 = subsample_sizes

    statuses = n00 * [[0, 0]] + n01 * [[0, 1]] + n10 * [[1, 0]] + n11 * [[1, 1]]
    statuses = np.array(statuses)

    outcomes = [
        correlated_binary_outcomes(n, (prob_eff, prob_tox), psi)
        for (n, (prob_eff, prob_tox, psi)) in zip(subsample_sizes, params)
    ]
    outcomes = np.vstack(outcomes)

    to_return = np.hstack([statuses, outcomes])
    np.random.shuffle(to_return)
    return to_return, (n00, n01, n10, n11)


def simulate_peps2_trial_batch(
    model,
    num_patients,
    prob_pretreated,
    prob_biomarker,
    prob_effes,
    prob_toxes,
    efftox_ors,
    num_batches,
    num_sims_per_batch,
    out_file=None,
):
    """Simulates a batch of PePS2 trials.

    Args:
        model (PePS2BeBOP): The trial model object.
        num_patients (int): The number of patients per trial.
        prob_pretreated (float): The probability of being pre-treated.
        prob_biomarker (float): The probability of being biomarker positive.
        prob_effes (list[float]): A list of efficacy probabilities for each
            subgroup.
        prob_toxes (list[float]): A list of toxicity probabilities for each
            subgroup.
        efftox_ors (list[float]): A list of efficacy-toxicity odds ratios.
        num_batches (int): The number of batches to simulate.
        num_sims_per_batch (int): The number of simulations per batch.
        out_file (str, optional): The output file to save the results.
            Defaults to `None`.

    Returns:
        collections.OrderedDict: A dictionary containing the simulation
            parameters and results.
    """
    sims = []
    sims_object = {}
    for i in range(num_batches):
        these_sims = simulate_peps2_trial(
            model,
            num_patients=num_patients,
            prob_pretreated=prob_pretreated,
            prob_biomarker=prob_biomarker,
            prob_effes=prob_effes,
            prob_toxes=prob_toxes,
            efftox_ors=efftox_ors,
            num_sims=num_sims_per_batch,
            log_every=0,
        )
        sims = sims + these_sims
        logger.info("Ran batch %s %s", i, datetime.datetime.now())

        sims_object = OrderedDict()
        sims_object["Parameters"] = peps2_parameters_report(
            num_patients=num_patients,
            prob_pretreated=prob_pretreated,
            prob_biomarker=prob_biomarker,
            prob_effes=prob_effes,
            prob_toxes=prob_toxes,
            efftox_ors=efftox_ors,
        )
        sims_object["Simulations"] = sims

        if out_file:
            try:
                json.dump(sims_object, open(out_file, "w"))
            except:
                import sys

                e = sys.exc_info()[0]
                logging.error(e.message)

    return sims_object


def get_corr(x):
    """Calculates the correlation between two columns of a matrix.

    Args:
        x (numpy.ndarray): A 2D numpy array.

    Returns:
        float: The correlation coefficient.
    """
    return np.corrcoef(x[:, 0], x[:, 1])[0, 1]


def peps2_parameters_report(
    num_patients, prob_pretreated, prob_biomarker, prob_effes, prob_toxes, efftox_ors
):
    """Generates a report of the PePS2 trial parameters.

    Args:
        num_patients (int): The number of patients.
        prob_pretreated (float): The probability of being pre-treated.
        prob_biomarker (float): The probability of being biomarker positive.
        prob_effes (list[float]): The efficacy probabilities.
        prob_toxes (list[float]): The toxicity probabilities.
        efftox_ors (list[float]): The efficacy-toxicity odds ratios.

    Returns:
        collections.OrderedDict: A dictionary with the parameter report.
    """
    parameters = OrderedDict()
    parameters["NumPatients"] = atomic_to_json(num_patients)
    parameters["Prob(Pretreated)"] = atomic_to_json(prob_pretreated)
    parameters["Prob(PD-L1+)"] = atomic_to_json(prob_biomarker)
    parameters["Prob(Efficacy)"] = iterable_to_json(prob_effes)
    parameters["Prob(Toxicity)"] = iterable_to_json(prob_toxes)
    parameters["Efficacy-Toxicity OR"] = iterable_to_json(efftox_ors)
    parameters["Groups"] = [
        "Not pretreated, PD-L1 negative",
        "Not pretreated, PD-L1 positive",
        "Pretreated, PD-L1 negative",
        "Pretreated, PD-L1 positive",
    ]
    pretreated_response_prob = prob_effes[3] * prob_biomarker + prob_effes[2] * (
        1 - prob_biomarker
    )
    parameters["Prob(Efficacy | Pretreated)"] = atomic_to_json(pretreated_response_prob)
    pretreated_response_odds = pretreated_response_prob / (1 - pretreated_response_prob)
    parameters["Odds(Efficacy | Pretreated)"] = pretreated_response_odds
    not_pretreated_response_prob = prob_effes[1] * prob_biomarker + prob_effes[0] * (
        1 - prob_biomarker
    )
    parameters["Prob(Efficacy | !Pretreated"] = not_pretreated_response_prob
    not_pretreated_response_odds = not_pretreated_response_prob / (
        1 - not_pretreated_response_prob
    )
    parameters["Odds(Efficacy | !Pretreated)"] = not_pretreated_response_odds
    pretreated_response_or = pretreated_response_odds / not_pretreated_response_odds
    parameters["Efficacy OR for Pretreated"] = atomic_to_json(pretreated_response_or)
    biomarker_pos_response_prob = prob_effes[3] * prob_pretreated + prob_effes[1] * (
        1 - prob_pretreated
    )
    parameters["Prob(Efficacy | PD-L1+)"] = atomic_to_json(biomarker_pos_response_prob)
    biomarker_pos_response_odds = biomarker_pos_response_prob / (
        1 - biomarker_pos_response_prob
    )
    parameters["Odds(Efficacy | PD-L1+)"] = atomic_to_json(biomarker_pos_response_odds)
    biomarker_neg_response_prob = prob_effes[2] * prob_pretreated + prob_effes[0] * (
        1 - prob_pretreated
    )
    parameters["Prob(Efficacy | PD-L1-"] = atomic_to_json(biomarker_neg_response_prob)
    biomarker_neg_response_odds = biomarker_neg_response_prob / (
        1 - biomarker_neg_response_prob
    )
    parameters["Odds(Efficacy | PD-L1-)"] = atomic_to_json(biomarker_neg_response_odds)
    biomarker_response_or = biomarker_pos_response_odds / biomarker_neg_response_odds
    parameters["Efficacy OR for PD-L1 +vs-"] = atomic_to_json(biomarker_response_or)

    return parameters


def simulate_peps2_trial(
    model,
    num_patients,
    prob_pretreated,
    prob_biomarker,
    prob_effes,
    prob_toxes,
    efftox_ors,
    num_sims=5,
    log_every=0,
):
    """Simulates a single PePS2 trial.

    Args:
        model (PePS2BeBOP): The trial model object.
        num_patients (int): The number of patients.
        prob_pretreated (float): The probability of being pre-treated.
        prob_biomarker (float): The probability of being biomarker positive.
        prob_effes (list[float]): The efficacy probabilities.
        prob_toxes (list[float]): The toxicity probabilities.
        efftox_ors (list[float]): The efficacy-toxicity odds ratios.
        num_sims (int, optional): The number of simulations to run.
            Defaults to 5.
        log_every (int, optional): The logging frequency. Defaults to 0.

    Returns:
        list: A list of simulation output dictionaries.
    """
    sims = []
    for i in range(num_sims):

        if log_every > 0 and i % log_every == 0:
            logger.debug("Iteration %s %s", i, datetime.datetime.now())

        sim_output = OrderedDict()

        x, n = simulate_peps2_patients(
            num_patients,
            prob_pretreated,
            prob_biomarker,
            zip(prob_effes, prob_toxes, efftox_ors),
        )
        sim_output["PreTreated"] = [int(y) for y in x[:, 0]]
        sim_output["PD-L1+"] = [int(y) for y in x[:, 1]]
        sim_output["Efficacy"] = [int(y) for y in x[:, 2]]
        sim_output["Toxicity"] = [int(y) for y in x[:, 3]]
        df = pd.DataFrame(x, columns=["PreTreated", "Mutated", "Eff", "Tox"])
        df["One"] = 1
        grouped = df.groupby(["PreTreated", "Mutated"])
        sub_df = grouped["One"].agg(np.sum)
        num_pats = [
            sub_df.get((0, 0), default=0),
            sub_df.get((0, 1), default=0),
            sub_df.get((1, 0), default=0),
            sub_df.get((1, 1), default=0),
        ]
        sim_output["GroupSizes"] = num_pats
        sub_df = grouped["Eff"].agg(np.sum)
        num_effs = [
            int(sub_df.get((0, 0), default=0)),
            int(sub_df.get((0, 1), default=0)),
            int(sub_df.get((1, 0), default=0)),
            int(sub_df.get((1, 1), default=0)),
        ]
        sim_output["GroupEfficacies"] = num_effs
        sub_df = grouped["Tox"].agg(np.sum)
        num_toxs = [
            int(sub_df.get((0, 0), default=0)),
            int(sub_df.get((0, 1), default=0)),
            int(sub_df.get((1, 0), default=0)),
            int(sub_df.get((1, 1), default=0)),
        ]
        sim_output["GroupToxicities"] = num_toxs

        model.reset()
        model.update(x)
        bebop_output = OrderedDict()
        bebop_output["ProbEff"] = iterable_to_json(np.round(model.prob_eff, 4))
        bebop_output["ProbTox"] = iterable_to_json(np.round(model.prob_tox, 4))
        bebop_output["ProbAccEff"] = iterable_to_json(np.round(model.prob_acc_eff, 4))
        bebop_output["ProbAccTox"] = iterable_to_json(np.round(model.prob_acc_tox, 4))

        bebop_output["Efficacy OR for Pretreated"] = iterable_to_json(
            np.round(model.efficacy_effect(1), 4)
        )
        bebop_output["Efficacy OR for PD-L1 +vs-"] = iterable_to_json(
            np.round(model.efficacy_effect(2), 4)
        )

        sim_output["BeBOP"] = bebop_output

        sims.append(sim_output)

    return sims


def splice_sims(in_files_pattern, out_file=None):
    """Splices simulation results from multiple files.

    Args:
        in_files_pattern (str): A glob pattern for the input files.
        out_file (str, optional): The output file to save the spliced
            results. Defaults to `None`.

    Returns:
        dict or None: The spliced simulation results, or `None` if no files
            are found.
    """
    def _splice_sims(sims1, sims2):
        sims1["NumSims"] = atomic_to_json(sims1["NumSims"] + sims2["NumSims"])
        sims1["Group Sizes"] = [
            iterable_to_json(x)
            for x in np.vstack((sims1["Group Sizes"], sims2["Group Sizes"]))
        ]
        sims1["Efficacy Events"] = [
            iterable_to_json(x)
            for x in np.vstack((sims1["Efficacy Events"], sims2["Efficacy Events"]))
        ]
        sims1["Toxicity Events"] = [
            iterable_to_json(x)
            for x in np.vstack((sims1["Toxicity Events"], sims2["Toxicity Events"]))
        ]

        sims1["BBB"]["Decisions"] = [
            iterable_to_json(x)
            for x in np.vstack((sims1["BBB"]["Decisions"], sims2["BBB"]["Decisions"]))
        ]
        sims1["BBB"]["Response OR for Biomarker Positive"] = [
            iterable_to_json(x)
            for x in np.vstack(
                (
                    sims1["BBB"]["Response OR for Biomarker Positive"],
                    sims2["BBB"]["Response OR for Biomarker Positive"],
                )
            )
        ]
        sims1["BBB"]["Response OR for Pretreated"] = [
            iterable_to_json(x)
            for x in np.vstack(
                (
                    sims1["BBB"]["Response OR for Pretreated"],
                    sims2["BBB"]["Response OR for Pretreated"],
                )
            )
        ]

        sims1["B&D+ChiSqu"]["Decisions"] = [
            iterable_to_json(x)
            for x in np.vstack(
                (sims1["B&D+ChiSqu"]["Decisions"], sims2["B&D+ChiSqu"]["Decisions"])
            )
        ]
        sims1["B&D+ChiSqu"]["Response OR for Biomarker Positive"] = [
            iterable_to_json(x)
            for x in np.vstack(
                (
                    sims1["B&D+ChiSqu"]["Response OR for Biomarker Positive"],
                    sims2["B&D+ChiSqu"]["Response OR for Biomarker Positive"],
                )
            )
        ]
        sims1["B&D+ChiSqu"]["Response OR for Pretreated"] = [
            iterable_to_json(x)
            for x in np.vstack(
                (
                    sims1["B&D+ChiSqu"]["Response OR for Pretreated"],
                    sims2["B&D+ChiSqu"]["Response OR for Pretreated"],
                )
            )
        ]
        return sims1

    files = glob.glob(in_files_pattern)
    if len(files):

        sims = json.load(open(files[0]))
        logger.info("Fetched from %s", files[0])
        for f in files[1:]:
            sub_sims = json.load(open(f))
            logger.info("Fetched from %s", f)
            sims = _splice_sims(sims, sub_sims)

    if out_file:
        json.dump(sims, open(out_file, "w"))
    else:
        return sims


from itertools import product


def tell_me_about_results(
    sims,
    eff_certainty_schemas=[[0.8] * 4, [0.7] * 4],
    tox_certainty_schemas=[[0.8] * 4, [0.7] * 4],
):
    """Prints a summary of the simulation results.

    Args:
        sims (dict): The simulation results object.
        eff_certainty_schemas (list[list[float]], optional): A list of
            efficacy certainty schemas.
        tox_certainty_schemas (list[list[float]], optional): A list of
            toxicity certainty schemas.
    """
    pretreated_efficacy_or = sims["Parameters"]["Efficacy OR for Pretreated"]
    pdl1_efficacy_or = sims["Parameters"]["Efficacy OR for PD-L1 +vs-"]

    num_sims = len(sims["Simulations"])
    n_by_group = reduce(
        lambda x, y: np.array(x) + np.array(y),
        map(lambda x: x["GroupSizes"], sims["Simulations"]),
    )
    eff_by_group = reduce(
        lambda x, y: np.array(x) + np.array(y),
        map(lambda x: x["GroupEfficacies"], sims["Simulations"]),
    )
    tox_by_group = reduce(
        lambda x, y: np.array(x) + np.array(y),
        map(lambda x: x["GroupToxicities"], sims["Simulations"]),
    )
    pretreated_efficacy_or_ci = np.array(
        map(lambda x: x["BeBOP"]["Efficacy OR for Pretreated"], sims["Simulations"])
    )
    pdl1_efficacy_or_ci = np.array(
        map(lambda x: x["BeBOP"]["Efficacy OR for PD-L1 +vs-"], sims["Simulations"])
    )

    logger.info("Params:")
    logger.info("Prob(Eff): %s", sims["Parameters"]["Prob(Efficacy)"])
    logger.info("Prob(Tox): %s", sims["Parameters"]["Prob(Toxicity)"])
    logger.info("Eff-Tox OR: %s", sims["Parameters"]["Efficacy-Toxicity OR"])
    logger.info("Prob(PreTreated): %s", sims["Parameters"]["Prob(Pretreated)"])
    logger.info("Prob(PD-L1+): %s", sims["Parameters"]["Prob(PD-L1+)"])
    logger.info("")
    logger.info("NumSims: %s", num_sims)
    logger.info("")
    logger.info("Events:")
    logger.info("Efficacy %: %s", 1.0 * sum(eff_by_group) / sum(n_by_group))
    logger.info("Toxicity %: %s", 1.0 * sum(tox_by_group) / sum(n_by_group))
    logger.info("")
    logger.info("By Group:")
    logger.info("Size: %s", 1.0 * n_by_group / num_sims)
    logger.info("Efficacies: %s", 1.0 * eff_by_group / num_sims)
    logger.info("Efficacy %: %s", 1.0 * eff_by_group / n_by_group)
    logger.info("Toxicities: %s", 1.0 * tox_by_group / num_sims)
    logger.info("Toxicity %: %s", 1.0 * tox_by_group / n_by_group)
    logger.info("")
    logger.info("")
    logger.info("BeBOP:")
    logger.info("")
    logger.info("Posterior:")
    logger.info(
        "Prob(Eff): %s",
        np.array(list(map(lambda x: x["BeBOP"]["ProbEff"], sims["Simulations"]))).mean(
            axis=0
        ),
    )
    logger.info(
        "Prob(AccEff): %s",
        np.array(
            list(map(lambda x: x["BeBOP"]["ProbAccEff"], sims["Simulations"]))
        ).mean(axis=0),
    )
    logger.info(
        "Prob(Tox): %s",
        np.array(list(map(lambda x: x["BeBOP"]["ProbTox"], sims["Simulations"]))).mean(
            axis=0
        ),
    )
    logger.info(
        "Prob(AccTox): %s",
        np.array(
            list(map(lambda x: x["BeBOP"]["ProbAccTox"], sims["Simulations"]))
        ).mean(axis=0),
    )
    logger.info("")
    logger.info("Approve Treatment:")
    for eff_certainty, tox_certainty in product(
        eff_certainty_schemas, tox_certainty_schemas
    ):
        accept_eff = np.array(
            map(lambda x: x["BeBOP"]["ProbAccEff"], sims["Simulations"])
        ) > np.array(eff_certainty)
        accept_tox = np.array(
            map(lambda x: x["BeBOP"]["ProbAccTox"], sims["Simulations"])
        ) > np.array(tox_certainty)
        logger.info(
            "%s %s %s",
            eff_certainty,
            tox_certainty,
            (accept_eff & accept_tox).mean(axis=0),
        )
    logger.info("")
    logger.info("BeBOP Coverage:")
    logger.info("Pre-Treated:")
    logger.info("True OR: %s", pretreated_efficacy_or)
    logger.info(
        "Coverage: %s",
        np.mean(
            list(
                map(
                    lambda x: (pretreated_efficacy_or > x[0])
                    and (pretreated_efficacy_or < x[2]),
                    pretreated_efficacy_or_ci,
                )
            )
        ),
    )
    logger.info("PD-L1:")
    logger.info("True OR: %s", pdl1_efficacy_or)
    logger.info(
        "Coverage: %s",
        np.mean(
            list(
                map(
                    lambda x: (pdl1_efficacy_or > x[0]) and (pdl1_efficacy_or < x[2]),
                    pdl1_efficacy_or_ci,
                )
            )
        ),
    )
