"""
Brock & Yap's novel seamless phase I/II efficacy/toxicity design, fusing
elements of Wages & Tait's design with elements of Thall & Cook's EffTox
design.
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


from random import sample

import numpy as np
from scipy.stats import beta, norm

from clintrials.core.math import empiric, inverse_empiric
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial
from clintrials.dosefinding.efftox import solve_metrizable_efftox_scenario
from clintrials.dosefinding.wagestait import _get_post_eff_bayes, _wt_get_theta_hat


class WATU(EfficacyToxicityDoseFindingTrial):
    """Brock & Yap's fusion of Wages & Tait's phase I/II design with Thall &
    Cook's EffTox utility contours.
    """

    def __init__(
        self,
        skeletons,
        prior_tox_probs,
        tox_target,
        tox_limit,
        eff_limit,
        metric,
        first_dose,
        max_size,
        stage_one_size=0,
        F_func=empiric,
        inverse_F=inverse_empiric,
        theta_prior=norm(0, np.sqrt(1.34)),
        beta_prior=norm(0, np.sqrt(1.34)),
        tox_certainty=0.05,
        eff_certainty=0.05,
        model_prior_weights=None,
        use_quick_integration=True,
        estimate_var=True,
        avoid_skipping_untried_escalation_stage_1=True,
        avoid_skipping_untried_deescalation_stage_1=True,
        avoid_skipping_untried_escalation_stage_2=True,
        avoid_skipping_untried_deescalation_stage_2=True,
        plugin_mean=False,
        mc_sample_size=10**5,
    ):
        """Initializes a WATU trial object.

        Args:
            skeletons (list[list[float]]): A list of efficacy skeletons.
            prior_tox_probs (list[float]): A list of prior toxicity
                probabilities.
            tox_target (float): The target toxicity rate.
            tox_limit (float): The maximum acceptable toxicity probability.
            eff_limit (float): The minimum acceptable efficacy probability.
            metric (LpNormCurve | InverseQuadraticCurve): An object for
                calculating the utility of efficacy-toxicity pairs.
            first_dose (int): The starting dose level (1-based).
            max_size (int): The maximum number of patients in the trial.
            stage_one_size (int, optional): The size of the first stage of the
                trial. Defaults to 0.
            F_func (callable, optional): The link function. Defaults to `empiric`.
            inverse_F (callable, optional): The inverse link function. Defaults
                to `inverse_empiric`.
            theta_prior (scipy.stats.rv_continuous, optional): The prior for
                theta. Defaults to a normal distribution.
            beta_prior (scipy.stats.rv_continuous, optional): The prior for beta.
                Defaults to a normal distribution.
            tox_certainty (float, optional): The posterior certainty required
                that toxicity is less than the cutoff. Defaults to 0.05.
            eff_certainty (float, optional): The posterior certainty required
                that efficacy is greater than the cutoff. Defaults to 0.05.
            model_prior_weights (list[float], optional): The prior weights for
                each model. Defaults to uniform weights.
            use_quick_integration (bool, optional): If `True`, uses a faster
                integration method. Defaults to `True`.
            estimate_var (bool, optional): If `True`, estimates the posterior
                variance of beta and theta. Defaults to `True`.
            avoid_skipping_untried_escalation_stage_1 (bool, optional): If
                `True`, avoids skipping untried doses in escalation in stage 1.
                Defaults to `True`.
            avoid_skipping_untried_deescalation_stage_1 (bool, optional): If
                `True`, avoids skipping untried doses in de-escalation in stage
                1. Defaults to `True`.
            avoid_skipping_untried_escalation_stage_2 (bool, optional): If
                `True`, avoids skipping untried doses in escalation in stage 2.
                Defaults to `True`.
            avoid_skipping_untried_deescalation_stage_2 (bool, optional): If
                `True`, avoids skipping untried doses in de-escalation in stage
                2. Defaults to `True`.
            plugin_mean (bool, optional): If `True`, estimates event curves by
                plugging the parameter estimate into the function. Defaults to
                `False`.
            mc_sample_size (int, optional): The number of samples to use in Monte
                Carlo estimation methods. Defaults to 10**5.

        Raises:
            ValueError: If the dimensions of the inputs are inconsistent.
        """
        EfficacyToxicityDoseFindingTrial.__init__(
            self, first_dose, len(prior_tox_probs), max_size
        )

        self.skeletons = skeletons
        self.K, self.I = np.array(skeletons).shape
        if self.I != len(prior_tox_probs):
            raise ValueError("prior_tox_probs should have %s items." % self.I)
        self.prior_tox_probs = prior_tox_probs
        self.tox_target = tox_target
        self.tox_limit = tox_limit
        self.eff_limit = eff_limit
        self.metric = metric
        self.stage_one_size = stage_one_size
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.theta_prior = theta_prior
        self.beta_prior = beta_prior
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty
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
        self.avoid_skipping_untried_escalation_stage_1 = (
            avoid_skipping_untried_escalation_stage_1
        )
        self.avoid_skipping_untried_deescalation_stage_1 = (
            avoid_skipping_untried_deescalation_stage_1
        )
        self.avoid_skipping_untried_escalation_stage_2 = (
            avoid_skipping_untried_escalation_stage_2
        )
        self.avoid_skipping_untried_deescalation_stage_2 = (
            avoid_skipping_untried_deescalation_stage_2
        )
        self.plugin_mean = plugin_mean
        self.mc_sample_size = mc_sample_size

        self.most_likely_model_index = np.random.choice(
            np.array(range(self.K))[
                self.model_prior_weights == max(self.model_prior_weights)
            ],
            1,
        )[0]
        self.w = np.zeros(self.K)
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
            avoid_skipping_untried_escalation=avoid_skipping_untried_escalation_stage_1,
            avoid_skipping_untried_deescalation=avoid_skipping_untried_deescalation_stage_1,
            plugin_mean=plugin_mean,
        )
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)
        self.theta_vars = np.zeros(self.K)

        self.utility = []
        self.dose_allocation_mode = 0

    def model_theta_hat(self):
        """Gets the theta estimate for the most likely model.

        Returns:
            float: The theta estimate.
        """
        return self.theta_hats[self.most_likely_model_index]

    def model_theta_var(self):
        """Gets the theta variance for the most likely model.

        Returns:
            float: The theta variance.
        """
        return self.theta_vars[self.most_likely_model_index]

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
            estimate_var=True,
        )
        theta_hats, theta_vars, model_probs = zip(*integrals)
        self.theta_hats = theta_hats
        self.theta_vars = theta_vars
        w = self.model_prior_weights * model_probs
        self.w = w / sum(w)
        most_likely_model_index = np.argmax(w)
        self.most_likely_model_index = most_likely_model_index
        self.post_tox_probs = np.array(self.crm.prob_tox())
        if self.plugin_mean:
            self.post_eff_probs = empiric(
                self.skeletons[most_likely_model_index],
                beta=theta_hats[most_likely_model_index],
            )
        else:
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

        if self.size() < self.stage_one_size:
            self._next_dose = self._stage_one_next_dose()
        else:
            self._next_dose = self._stage_two_next_dose(
                self.post_tox_probs, self.post_eff_probs
            )

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
        self.theta_vars = np.zeros(self.K)
        self.crm.reset()

    def has_more(self):
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def optimal_decision(self, prob_tox, prob_eff):
        admiss, u, u_star, obd, u_cushtion = solve_metrizable_efftox_scenario(
            prob_tox, prob_eff, self.metric, self.tox_limit, self.eff_limit
        )
        return obd

    def prob_eff_exceeds(self, eff_cutoff):
        """Calculates the probability that efficacy exceeds a cutoff.

        Args:
            eff_cutoff (float): The efficacy cutoff.

        Returns:
            numpy.ndarray: An array of probabilities for each dose level.
        """
        skeleton = np.array(self.skeletons[self.most_likely_model_index])
        probs = np.zeros_like(skeleton, dtype=float)

        if eff_cutoff >= 1.0:
            return probs

        if eff_cutoff < 0.0:
            return np.ones_like(skeleton, dtype=float)

        theta_posterior = norm(loc=self.model_theta_hat(), scale=np.sqrt(self.model_theta_var()))

        one_mask = skeleton == 1.0
        probs[one_mask] = 1.0

        zero_mask = skeleton == 0.0
        probs[zero_mask] = 0.0

        middle_mask = ~(one_mask | zero_mask)
        if np.any(middle_mask):
            sub_skeleton = skeleton[middle_mask]
            eff_cutoff_clipped = np.maximum(eff_cutoff, 1e-9)
            thresholds = np.log(eff_cutoff_clipped) / np.log(sub_skeleton)
            probs[middle_mask] = theta_posterior.cdf(thresholds)

        return probs

    def prob_acc_eff(self, threshold=None):
        """Calculates the probability of acceptable efficacy.

        Args:
            threshold (float, optional): The efficacy threshold. Defaults to
                `self.eff_limit`.

        Returns:
            numpy.ndarray: An array of probabilities for each dose level.
        """
        if threshold is None:
            threshold = self.eff_limit
        return self.prob_eff_exceeds(threshold)

    def prob_acc_tox(self, threshold=None, **kwargs):
        """Calculates the probability of acceptable toxicity.

        Args:
            threshold (float, optional): The toxicity threshold. Defaults to
                `self.tox_limit`.
            **kwargs: Additional arguments for `crm.prob_tox_exceeds`.

        Returns:
            numpy.ndarray: An array of probabilities for each dose level.
        """
        if threshold is None:
            threshold = self.tox_limit
        return 1 - self.crm.prob_tox_exceeds(threshold, **kwargs)

    def _stage_one_next_dose(self):
        prob_unacc_tox = self.crm.prob_tox_exceeds(
            self.tox_limit, n=self.mc_sample_size
        )
        prob_unacc_eff = 1 - self.prob_eff_exceeds(self.eff_limit)
        admissable = [
            (prob_tox < (1 - self.tox_certainty))
            and (prob_eff < (1 - self.eff_certainty))
            for (prob_eff, prob_tox) in zip(prob_unacc_eff, prob_unacc_tox)
        ]
        admissable_set = [i + 1 for i, x in enumerate(admissable) if x]
        self._admissable_set = admissable_set

        if self.size() > 0:
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            attractiveness = np.abs(
                np.array(self.crm.prob_tox()) - self.tox_target
            )
            for i in np.argsort(
                attractiveness
            ):
                dose_level = i + 1
                if dose_level in admissable_set:
                    if (
                        self.avoid_skipping_untried_escalation_stage_1
                        and max_dose_given
                        and dose_level - max_dose_given > 1
                    ):
                        pass
                    elif (
                        self.avoid_skipping_untried_deescalation_stage_1
                        and min_dose_given
                        and min_dose_given - dose_level > 1
                    ):
                        pass
                    else:
                        self._status = 1
                        self._next_dose = dose_level
                        break
            else:
                self._next_dose = -1
                self._status = -1
        else:
            self._next_dose = self.first_dose()
            self._status = -10

        return self._next_dose

    def _stage_two_next_dose(self, tox_probs, eff_probs):
        prob_unacc_tox = self.crm.prob_tox_exceeds(
            self.tox_limit, n=self.mc_sample_size
        )
        prob_unacc_eff = 1 - self.prob_eff_exceeds(self.eff_limit)
        admissable = [
            (prob_tox < (1 - self.tox_certainty))
            and (prob_eff < (1 - self.eff_certainty))
            for (prob_eff, prob_tox) in zip(prob_unacc_eff, prob_unacc_tox)
        ]
        admissable_set = [i + 1 for i, x in enumerate(admissable) if x]
        self._admissable_set = admissable_set
        utility = np.array([self.metric(x[0], x[1]) for x in zip(eff_probs, tox_probs)])
        self.utility = utility

        if self.size() > 0:
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            for i in np.argsort(
                -utility
            ):
                dose_level = i + 1
                if dose_level in admissable_set:
                    if (
                        self.avoid_skipping_untried_escalation_stage_2
                        and max_dose_given
                        and dose_level - max_dose_given > 1
                    ):
                        pass
                    elif (
                        self.avoid_skipping_untried_deescalation_stage_2
                        and min_dose_given
                        and min_dose_given - dose_level > 1
                    ):
                        pass
                    else:
                        self._status = 1
                        self._next_dose = dose_level
                        break
            else:
                self._next_dose = -1
                self._status = -1
        else:
            self._next_dose = self.first_dose()
            self._status = -10

        return self._next_dose
