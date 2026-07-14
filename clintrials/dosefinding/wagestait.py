from __future__ import annotations
from typing import Any, Sequence
import numpy as np
"""
An implementation of Wages & Tait's adaptive Bayesian design for dose-finding
in clinical trials.


Random Seed Strategy: {wagestait_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


from scipy.stats import beta, norm

from clintrials.core.errors import ErrorTemplates
from clintrials.core.math import empiric, inverse_empiric, bernoulli_likelihood
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial

_min_theta, _max_theta = -10, 10


def _wt_lik(cases: Any, skeleton: Any, theta: Any, F: Any = empiric, a0: Any = 0) -> Any:
    """Calculates the compound likelihood for the Wages & Tait method.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases, where each
            case is a tuple of (dose, toxicity, efficacy).
        skeleton (list[float]): A list of prior efficacy probabilities.
        theta (float): The slope parameter.
        F (callable, optional): The link function. Defaults to `empiric`.
        a0 (float, optional): The intercept parameter. Defaults to 0.

    Returns:
        float: The compound likelihood.
    """
    l = 1
    for dose, tox, eff in cases:
        p = F(skeleton[dose - 1], a0=a0, beta=theta)
        l = l * bernoulli_likelihood(p, eff, log=False)
    return l


def _wt_log_lik(cases: Any, skeleton: Any, theta: Any, F: Any = empiric, a0: Any = 0) -> Any:
    """Calculates the compound log-likelihood for the Wages & Tait method.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases, where each
            case is a tuple of (dose, toxicity, efficacy).
        skeleton (list[float]): A list of prior efficacy probabilities.
        theta (float): The slope parameter.
        F (callable, optional): The link function. Defaults to `empiric`.
        a0 (float, optional): The intercept parameter. Defaults to 0.

    Returns:
        float: The compound log-likelihood.
    """
    ll = 0
    for dose, tox, eff in cases:
        p = F(skeleton[dose - 1], a0=a0, beta=theta)
        ll += bernoulli_likelihood(p, eff, log=True)
    return ll


def _wt_get_theta_hat(cases: Any, skeletons: Any, theta_prior: Any, F: Any = empiric, use_quick_integration: Any = False, estimate_var: Any = False) -> Any:
    """Estimates the theta parameter for the Wages & Tait method.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases.
        skeletons (list[list[float]]): A list of efficacy skeletons.
        theta_prior (scipy.stats.rv_continuous): The prior distribution for
            theta.
        F (callable, optional): The link function. Defaults to `empiric`.
        use_quick_integration (bool, optional): Ignored. Included for backward compatibility.
        estimate_var (bool, optional): If `True`, estimates the posterior
            variance of theta. Defaults to `False`.

    Returns:
        list[tuple[float, float | None, float]]: A list of tuples, where each
            tuple contains the posterior mean of theta, its variance (or
            `None`), and the model probability.
    """
    from clintrials.core.numerics import integrate_posterior_1d

    theta_hats = []
    for skeleton in skeletons:

        def logpost(t: Any) -> Any:
            ll = _wt_log_lik(cases, skeleton, t, F)
            return ll + np.log(theta_prior.pdf(t) + 1e-300)

        theta_hat, diag = integrate_posterior_1d(  # type: ignore
            logpost, lambda t: t, _min_theta, _max_theta, return_diagnostics=True
        )
        marginal_likelihood = diag["log_marginal"]

        if estimate_var:
            exp_x2 = integrate_posterior_1d(  # type: ignore
                logpost, lambda t: t**2, _min_theta, _max_theta
            )
            var = exp_x2 - theta_hat**2
            theta_hats.append((theta_hat, var, marginal_likelihood))
        else:
            theta_hats.append((theta_hat, None, marginal_likelihood))
    return theta_hats


def _get_post_eff_bayes(cases: Any, skeleton: Any, dose_labels: Any, theta_prior: Any, F: Any = empiric, use_quick_integration: Any = False) -> Any:
    """Calculates the posterior probability of efficacy using Bayesian integration.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases.
        skeleton (list[float]): A list of prior efficacy probabilities.
        dose_labels (list[float]): The dose labels.
        theta_prior (scipy.stats.rv_continuous): The prior for theta.
        F (callable, optional): The link function. Defaults to `empiric`.
        use_quick_integration (bool, optional): Ignored. Included for backward compatibility.

    Returns:
        numpy.ndarray: An array of posterior probabilities of efficacy.
    """
    from clintrials.core.numerics import integrate_posterior_1d

    def logpost(t: Any) -> Any:
        ll = _wt_log_lik(cases, skeleton, t, F)
        return ll + np.log(theta_prior.pdf(t) + 1e-300)

    post_eff = []
    intercept = 0
    for x in dose_labels:
        prob = integrate_posterior_1d(  # type: ignore
            logpost, lambda t: F(x, a0=intercept, beta=t), _min_theta, _max_theta
        )
        post_eff.append(prob)

    return np.array(post_eff)


class WagesTait(EfficacyToxicityDoseFindingTrial):
    """An object-oriented implementation of the Wages & Tait adaptive Phase I/II
    design for oncology trials of molecularly targeted agents.
    """

    def __init__(self, skeletons: Any, prior_tox_probs: Any, tox_target: Any, tox_limit: Any, eff_limit: Any, first_dose: Any, max_size: Any, randomisation_stage_size: Any, F_func: Any = empiric, inverse_F: Any = inverse_empiric, theta_prior: Any = norm(0, np.sqrt(1.34)), beta_prior: Any = norm(0, np.sqrt(1.34)), excess_toxicity_alpha: Any = 0.025, deficient_efficacy_alpha: Any = 0.025, model_prior_weights: Any = None, use_quick_integration: Any = False, estimate_var: Any = False) -> None:
        """Initializes a WagesTait trial object.

        Args:
            skeletons (list[list[float]]): A list of efficacy skeletons.
            prior_tox_probs (list[float]): A list of prior toxicity
                probabilities.
            tox_target (float): The target toxicity rate.
            tox_limit (float): The maximum acceptable toxicity probability.
            eff_limit (float): The minimum acceptable efficacy probability.
            first_dose (int): The starting dose level (1-based).
            max_size (int): The maximum number of patients in the trial.
            randomisation_stage_size (int): The number of patients to randomize
                in the first stage.
            F_func (callable, optional): The link function. Defaults to `empiric`.
            inverse_F (callable, optional): The inverse link function. Defaults
                to `inverse_empiric`.
            theta_prior (scipy.stats.rv_continuous, optional): The prior for
                theta. Defaults to a normal distribution.
            beta_prior (scipy.stats.rv_continuous, optional): The prior for beta.
                Defaults to a normal distribution.
            excess_toxicity_alpha (float, optional): The significance level for
                the excess toxicity test. Defaults to 0.025.
            deficient_efficacy_alpha (float, optional): The significance level
                for the deficient efficacy test. Defaults to 0.025.
            model_prior_weights (list[float], optional): The prior weights for
                each model. Defaults to uniform weights.
            use_quick_integration (bool, optional): If `True`, uses a faster
                integration method. Defaults to `False`.
            estimate_var (bool, optional): If `True`, estimates the posterior
                variance of beta and theta. Defaults to `False`.

        Raises:
            ValueError: If the dimensions of the inputs are inconsistent, or
                if `tox_target` is greater than `tox_limit`.
        """
        EfficacyToxicityDoseFindingTrial.__init__(
            self, first_dose, len(prior_tox_probs), max_size
        )

        self.skeletons = skeletons
        self.K, self.I = np.array(skeletons).shape  # type: ignore
        if self.I != len(prior_tox_probs):
            raise ValueError(ErrorTemplates.EXPECTED_LENGTH.format(name="prior_tox_probs", expected_length=self.I))
        if tox_target > tox_limit:
            raise ValueError(ErrorTemplates.LE.format(name="tox_target", bound="tox_limit"))
        self.prior_tox_probs = np.array(prior_tox_probs)
        self.tox_limit = tox_limit
        self.eff_limit = eff_limit
        self.randomisation_stage_size = randomisation_stage_size
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.theta_prior = theta_prior
        self.beta_prior = beta_prior
        self.excess_toxicity_alpha = excess_toxicity_alpha
        self.deficient_efficacy_alpha = deficient_efficacy_alpha
        if model_prior_weights is not None:
            if self.K != len(model_prior_weights):
                raise ValueError(ErrorTemplates.EXPECTED_LENGTH.format(name="model_prior_weights", expected_length=self.K))
            if sum(model_prior_weights) == 0:
                raise ValueError("model_prior_weights cannot sum to zero.")
            self.model_prior_weights = model_prior_weights / sum(model_prior_weights)
        else:
            self.model_prior_weights = np.ones(self.K) / self.K
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var

        self.most_likely_model_index = self.rng.choice(  # type: ignore
            np.array(range(self.K))[  # type: ignore
                self.model_prior_weights == max(self.model_prior_weights)
            ],
            1,
        )[0]
        self.w = np.zeros(self.K)
        if first_dose is None:
            self._next_dose = self._randomise_next_dose(
                prior_tox_probs, self.skeletons[self.most_likely_model_index]
            )
            self.randomise_at_start = True
        else:
            self.randomise_at_start = False
        self.crm = CRM(  # type: ignore
            prior=prior_tox_probs,
            target=tox_target,
            first_dose=first_dose,
            max_size=max_size,
            F_func=empiric,
            inverse_F=inverse_empiric,
            beta_prior=beta_prior,
            use_quick_integration=use_quick_integration,
            estimate_var=estimate_var,
            plugin_mean=False,
        )
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)

    def set_rng(self, rng: Any) -> None:
        """Inject a local RNG generator for reproducible, state-free random generation."""
        super().set_rng(rng)
        if hasattr(self, "crm"):
            self.crm.set_rng(rng)

    def dose_toxicity_lower_bound(self, dose_level: Any, alpha: Any = 0.025) -> Any:
        """Gets the lower bound of the toxicity probability for a dose level.

        This method uses the Clopper-Pearson (exact) method.

        Args:
            dose_level (int): The 1-based dose level.
            alpha (float, optional): The significance level. Defaults to 0.025.

        Returns:
            float: The lower bound of the toxicity probability.
        """
        if 0 < dose_level <= len(self.post_tox_probs):
            n = self.treated_at_dose(dose_level)
            x = self.toxicities_at_dose(dose_level)
            if n > 0:
                return beta(x, n - x + 1).ppf(alpha)
        return 0

    def dose_efficacy_upper_bound(self, dose_level: Any, alpha: Any = 0.025) -> Any:
        """Gets the upper bound of the efficacy probability for a dose level.

        This method uses the Clopper-Pearson (exact) method.

        Args:
            dose_level (int): The 1-based dose level.
            alpha (float, optional): The significance level. Defaults to 0.025.

        Returns:
            float: The upper bound of the efficacy probability.
        """
        if 0 < dose_level <= len(self.post_eff_probs):
            n = self.treated_at_dose(dose_level)
            x = self.efficacies_at_dose(dose_level)
            if n > 0:
                return beta(x + 1, n - x).ppf(1 - alpha)
        return 1

    def model_theta_hat(self) -> Any:
        """Gets the theta estimate for the most likely model.

        Returns:
            float: The theta estimate.
        """
        return self.theta_hats[self.most_likely_model_index]  # type: ignore

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self) -> Any:
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        toxicity_cases = []
        for dose, tox, eff in cases:
            toxicity_cases.append((dose, tox))
        self.crm.reset()
        self.crm.update(toxicity_cases)

        integrals = _wt_get_theta_hat(
            cases,
            self.skeletons,
            self.theta_prior,
            use_quick_integration=self.use_quick_integration,
            estimate_var=False,
        )
        theta_hats, theta_vars, log_marginal = zip(*integrals)

        self.theta_hats = theta_hats
        log_w = np.log(self.model_prior_weights + 1e-300) + np.array(log_marginal)
        log_w = log_w - np.max(log_w)
        w = np.exp(log_w)
        self.w = w / sum(w)
        most_likely_model_index = np.argmax(self.w)
        self.most_likely_model_index = most_likely_model_index
        self.post_tox_probs = np.array(self.crm.prob_tox())
        a0 = 0
        theta0 = self.theta_prior.mean()
        dose_labels = [
            self.inverse_F(p, a0=a0, beta=theta0)
            for p in self.skeletons[most_likely_model_index]
        ]
        self.post_eff_probs = _get_post_eff_bayes(
            cases,
            self.skeletons[most_likely_model_index],
            dose_labels,
            self.theta_prior,
            use_quick_integration=self.use_quick_integration,
        )

        if self.size() < self.randomisation_stage_size:
            self._next_dose = self._randomise_next_dose(
                self.post_tox_probs, self.post_eff_probs
            )
        else:
            self._next_dose = self._maximise_next_dose(
                self.post_tox_probs, self.post_eff_probs
            )

        if (
            self.dose_toxicity_lower_bound(1, self.excess_toxicity_alpha)
            > self.tox_limit
        ):
            self._status = -3
            self._next_dose = -1
            self._admissable_set = []
        if self.size() >= self.randomisation_stage_size:
            if (
                self.dose_efficacy_upper_bound(
                    self._next_dose, self.deficient_efficacy_alpha
                )
                < self.eff_limit
            ):
                self._status = -4
                self._next_dose = -1
                self._admissable_set = []

        return self._next_dose

    def _EfficacyToxicityDoseFindingTrial__reset(self) -> Any:
        self.most_likely_model_index = self.rng.choice(
            np.array(range(self.K))[  # type: ignore
                self.model_prior_weights == max(self.model_prior_weights)
            ],
            1,
        )[0]
        self.w = np.zeros(self.K)
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)
        self.crm.reset()
        if self.randomise_at_start:
            self._next_dose = self._randomise_next_dose(
                self.prior_tox_probs, self.skeletons[self.most_likely_model_index]
            )

    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def optimal_decision(self, prob_tox: Sequence[float], prob_eff: Sequence[float]) -> int:
        """Determines the optimal biological dose.

        Args:
            prob_tox (numpy.ndarray): The probability of toxicity for each dose.
            prob_eff (numpy.ndarray): The probability of efficacy for each dose.

        Returns:
            int: The optimal biological dose.
        """
        admiss = prob_tox <= self.tox_limit
        if sum(admiss) > 0:
            wt_obd = np.nanargmax(np.where(admiss, prob_eff, np.nan)) + 1
        else:
            wt_obd = -1  # type: ignore
        return wt_obd  # type: ignore

    def _randomise_next_dose(self, tox_probs: Any, eff_probs: Any) -> Any:
        acceptable_doses = tox_probs <= self.tox_limit
        if sum(acceptable_doses) > 0:
            prob_randomise = []
            for acc, eff in zip(acceptable_doses, eff_probs):
                if acc:
                    prob_randomise.append(eff)
                else:
                    prob_randomise.append(0)
            prob_randomise = np.array(prob_randomise) / sum(prob_randomise)
            self._status = 1
            self._admissable_set = [
                i
                for (acc, i) in zip(acceptable_doses, range(1, self.num_doses + 1))
                if acc
            ]
            return self.rng.choice(np.array(range(1, self.I + 1)), p=prob_randomise)
        else:
            self._status = -1
            self._admissable_set = []
            return -1

    def _maximise_next_dose(self, tox_probs: Any, eff_probs: Any) -> Any:
        acceptable_doses = tox_probs <= self.tox_limit
        if sum(acceptable_doses) > 0:
            self._status = 1
            self._admissable_set = [
                i
                for (acc, i) in zip(acceptable_doses, range(1, self.num_doses + 1))
                if acc
            ]
            return np.argmax(np.array(eff_probs)[acceptable_doses]) + 1  # type: ignore
        else:
            self._status = -1
            self._admissable_set = []
            return -1


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import REGISTRY
    __doc__ = __doc__.format(**REGISTRY)
