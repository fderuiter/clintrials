"""
An implementation of Wages & Tait's adaptive Bayesian design for dose-finding
in clinical trials.
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


from random import sample

import numpy as np
from numpy import trapezoid
from scipy.integrate import quad
from scipy.stats import beta, norm

from clintrials.core.math import empiric, inverse_empiric
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial

_min_theta, _max_theta = -10, 10


def _wt_lik(cases, skeleton, theta, F=empiric, a0=0):
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
        l = l * p**eff * (1 - p) ** (1 - eff)
    return l


def _wt_get_theta_hat(
    cases,
    skeletons,
    theta_prior,
    F=empiric,
    use_quick_integration=False,
    estimate_var=False,
):
    """Estimates the theta parameter for the Wages & Tait method.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases.
        skeletons (list[list[float]]): A list of efficacy skeletons.
        theta_prior (scipy.stats.rv_continuous): The prior distribution for
            theta.
        F (callable, optional): The link function. Defaults to `empiric`.
        use_quick_integration (bool, optional): If `True`, uses a faster but
            less accurate integration method. Defaults to `False`.
        estimate_var (bool, optional): If `True`, estimates the posterior
            variance of theta. Defaults to `False`.

    Returns:
        list[tuple[float, float | None, float]]: A list of tuples, where each
            tuple contains the posterior mean of theta, its variance (or
            `None`), and the model probability.
    """
    theta_hats = []
    for skeleton in skeletons:
        if use_quick_integration:
            n = int(
                100 * max(np.log(len(cases) + 1) / 2, 1)
            )
            z, dz = np.linspace(_min_theta, _max_theta, num=n, retstep=True)
            denom_y = _wt_lik(cases, skeleton, z, F) * theta_prior.pdf(z)
            num_y = z * denom_y
            num = trapezoid(num_y, z, dz)
            denom = trapezoid(denom_y, z, dz)
            theta_hat = num / denom
            if estimate_var:
                num2_y = z**2 * denom_y
                num2 = trapezoid(num2_y, z, dz)
                exp_x2 = num2 / denom
                var = exp_x2 - theta_hat**2
                theta_hats.append((theta_hat, var, denom))
            else:
                theta_hats.append((num / denom, None, denom))
        else:
            num = quad(
                lambda t: t * _wt_lik(cases, skeleton, t, F) * theta_prior.pdf(t),
                -np.inf,
                np.inf,
            )
            denom = quad(
                lambda t: _wt_lik(cases, skeleton, t, F) * theta_prior.pdf(t),
                -np.inf,
                np.inf,
            )
            theta_hat = num[0] / denom[0]
            if estimate_var:
                num2 = quad(
                    lambda t: t**2
                    * _wt_lik(cases, skeleton, t, F)
                    * theta_prior.pdf(t),
                    -np.inf,
                    np.inf,
                )
                exp_x2 = num2[0] / denom[0]
                var = exp_x2 - theta_hat**2
                theta_hats.append((theta_hat, var, denom[0]))
            else:
                theta_hats.append((theta_hat, None, denom[0]))
    return theta_hats


def _get_post_eff_bayes(
    cases, skeleton, dose_labels, theta_prior, F=empiric, use_quick_integration=False
):
    """Calculates the posterior probability of efficacy using Bayesian integration.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases.
        skeleton (list[float]): A list of prior efficacy probabilities.
        dose_labels (list[float]): The dose labels.
        theta_prior (scipy.stats.rv_continuous): The prior for theta.
        F (callable, optional): The link function. Defaults to `empiric`.
        use_quick_integration (bool, optional): If `True`, uses a faster
            integration method. Defaults to `False`.

    Returns:
        numpy.ndarray: An array of posterior probabilities of efficacy.
    """
    post_eff = []
    intercept = 0
    if use_quick_integration:
        n = int(
            100 * max(np.log(len(cases) + 1) / 2, 1)
        )
        z, dz = np.linspace(_min_theta, _max_theta, num=n, retstep=True)
        denom_y = _wt_lik(cases, skeleton, z, F) * theta_prior.pdf(z)
        denom = trapezoid(denom_y, z, dz)
        for x in dose_labels:
            num_y = F(x, a0=intercept, beta=z) * denom_y
            num = trapezoid(num_y, z, dz)
            post_eff.append(num / denom)
    else:
        denom = quad(
            lambda t: theta_prior.pdf(t) * _wt_lik(cases, skeleton, t, F),
            -np.inf,
            np.inf,
        )
        for x in dose_labels:
            num = quad(
                lambda t: F(x, a0=intercept, beta=t)
                * theta_prior.pdf(t)
                * _wt_lik(cases, skeleton, t, F),
                -np.inf,
                np.inf,
            )
            post_eff.append(num[0] / denom[0])

    return np.array(post_eff)


class WagesTait(EfficacyToxicityDoseFindingTrial):
    """An object-oriented implementation of the Wages & Tait adaptive Phase I/II
    design for oncology trials of molecularly targeted agents.
    """

    def __init__(
        self,
        skeletons,
        prior_tox_probs,
        tox_target,
        tox_limit,
        eff_limit,
        first_dose,
        max_size,
        randomisation_stage_size,
        F_func=empiric,
        inverse_F=inverse_empiric,
        theta_prior=norm(0, np.sqrt(1.34)),
        beta_prior=norm(0, np.sqrt(1.34)),
        excess_toxicity_alpha=0.025,
        deficient_efficacy_alpha=0.025,
        model_prior_weights=None,
        use_quick_integration=False,
        estimate_var=False,
    ):
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
        self.K, self.I = np.array(skeletons).shape
        if self.I != len(prior_tox_probs):
            raise ValueError("prior_tox_probs should have %s items." % self.I)
        if tox_target > tox_limit:
            raise ValueError(
                "tox_target is greater than tox_limit. That does not sound clever."
            )
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
                raise ValueError("model_prior_weights should have %s items." % self.K)
            if sum(model_prior_weights) == 0:
                raise ValueError("model_prior_weights cannot sum to zero.")
            self.model_prior_weights = model_prior_weights / sum(model_prior_weights)
        else:
            self.model_prior_weights = np.ones(self.K) / self.K
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var

        self.most_likely_model_index = np.random.choice(
            np.array(range(self.K))[
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
        self.crm = CRM(
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

    def dose_toxicity_lower_bound(self, dose_level, alpha=0.025):
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

    def dose_efficacy_upper_bound(self, dose_level, alpha=0.025):
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

    def model_theta_hat(self):
        """Gets the theta estimate for the most likely model.

        Returns:
            float: The theta estimate.
        """
        return self.theta_hats[self.most_likely_model_index]

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self):
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
        theta_hats, theta_vars, model_probs = zip(*integrals)

        self.theta_hats = theta_hats
        w = self.model_prior_weights * model_probs
        self.w = w / sum(w)
        most_likely_model_index = np.argmax(w)
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

    def _EfficacyToxicityDoseFindingTrial__reset(self):
        self.most_likely_model_index = sample(
            np.array(range(self.K))[
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

    def has_more(self):
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def optimal_decision(self, prob_tox, prob_eff):
        admiss = prob_tox <= self.tox_limit
        if sum(admiss) > 0:
            wt_obd = np.nanargmax(np.where(admiss, prob_eff, np.nan)) + 1
        else:
            wt_obd = -1
        return wt_obd

    def _randomise_next_dose(self, tox_probs, eff_probs):
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
            return np.random.choice(range(1, self.I + 1), p=prob_randomise)
        else:
            self._status = -1
            self._admissable_set = []
            return -1

    def _maximise_next_dose(self, tox_probs, eff_probs):
        acceptable_doses = tox_probs <= self.tox_limit
        if sum(acceptable_doses) > 0:
            self._status = 1
            self._admissable_set = [
                i
                for (acc, i) in zip(acceptable_doses, range(1, self.num_doses + 1))
                if acc
            ]
            return np.argmax(np.array(eff_probs)[acceptable_doses]) + 1
        else:
            self._status = -1
            self._admissable_set = []
            return -1
