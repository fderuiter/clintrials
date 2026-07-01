from __future__ import annotations
from typing import Any, Callable, Optional, Union, Sequence, Mapping, Dict, Tuple, List, Iterable
import numpy as np
import numpy.typing as npt
import pandas as pd
"""
Continual Reassessment Method (CRM) for dose-finding clinical trials.

Random Seed Strategy: {crm_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

import logging
import warnings
from collections import OrderedDict

import numpy as np
from clintrials.core.registry import REGISTRY, inject_docs
from numpy import trapezoid
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from clintrials.core.math import empiric, inverse_empiric, inverse_logistic, logistic
from clintrials.dosefinding import DoseFindingTrial
from clintrials.utils import atomic_to_json, iterable_to_json

_min_beta, _max_beta = -10, 10


def _toxicity_likelihood(link_func: Any, a0: Any, beta: Any, dose: Any, tox: Any, log: Any = False) -> Any:
    """Calculates the likelihood of a single toxicity outcome.

    Args:
        link_func (callable): The link function (e.g., logistic or empiric).
        a0 (float): The intercept parameter for the link function.
        beta (float): The slope parameter for the link function.
        dose (float): The dose level.
        tox (int): The toxicity outcome (1 for toxicity, 0 for no toxicity).
        log (bool, optional): If `True`, returns the log-likelihood.
            Defaults to `False`.

    Returns:
        float: The likelihood or log-likelihood of the toxicity outcome.
    """
    p = link_func(dose, a0, beta)
    if log:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return tox * np.log(p) + (1 - tox) * np.log(1 - p)
    else:
        return p**tox * (1 - p) ** (1 - tox)


def _compound_toxicity_likelihood(link_func: Any, a0: Any, beta: Any, doses: Any, toxs: Any, log: Any = False) -> Any:
    """Calculates the compound likelihood of multiple toxicity outcomes.

    Args:
        link_func (callable): The link function.
        a0 (float): The intercept parameter.
        beta (float): The slope parameter.
        doses (list[float]): A list of dose levels.
        toxs (list[int]): A list of toxicity outcomes.
        log (bool, optional): If `True`, returns the log-likelihood.
            Defaults to `False`.

    Returns:
        float: The compound likelihood or log-likelihood.
    """
    from clintrials.validation import validate_matching_lengths

    validate_matching_lengths(doses=doses, toxs=toxs)  # type: ignore

    if log:
        l = 0
        for dose, tox in zip(doses, toxs):
            l += _toxicity_likelihood(link_func, a0, beta, dose, tox, log=True)
        return l
    else:
        l = 1
        for dose, tox in zip(doses, toxs):
            l *= _toxicity_likelihood(link_func, a0, beta, dose, tox, log=False)
        return l


def _get_beta_hat_bayes(F: Any, intercept: Any, codified_doses_given: Any, toxs: Any, beta_pdf: Any, use_quick_integration: Any = False, estimate_var: Any = False) -> Any:
    """Estimates the beta parameter using Bayesian inference.

    Args:
        F (callable): The link function.
        intercept (float): The intercept parameter.
        codified_doses_given (list[float]): The codified dose levels given.
        toxs (list[int]): The observed toxicity events.
        beta_pdf (callable): The PDF of the prior distribution for beta.
        use_quick_integration (bool, optional): Ignored. Included for backward compatibility.
        estimate_var (bool, optional): If `True`, estimates the variance of
            beta. Defaults to `False`.

    Returns:
        tuple[float, float | None]: A tuple containing the posterior mean and
            variance of beta. The variance is `None` if `estimate_var` is
            `False`.
    """
    from clintrials.core.numerics import integrate_posterior_1d

    def logpost(t: Any) -> Any:
        ll = _compound_toxicity_likelihood(
            F, intercept, t, codified_doses_given, toxs, log=True
        )
        return ll + np.log(beta_pdf(t) + 1e-300)

    beta_hat = integrate_posterior_1d(logpost, lambda t: t, _min_beta, _max_beta)  # type: ignore

    if estimate_var:
        exp_x2 = integrate_posterior_1d(logpost, lambda t: t**2, _min_beta, _max_beta)  # type: ignore
        var = exp_x2 - beta_hat**2
    else:
        var = None

    return beta_hat, var


def _get_beta_hat_mle(F: Any, intercept: Any, codified_doses_given: Any, toxs: Any, estimate_var: Any = False) -> Any:
    """Estimates the beta parameter using maximum likelihood estimation (MLE).

    Args:
        F (callable): The link function.
        intercept (float): The intercept parameter.
        codified_doses_given (list[float]): The codified dose levels given.
        toxs (list[int]): The observed toxicity events.
        estimate_var (bool, optional): If `True`, estimates the variance of
            beta. Defaults to `False`.

    Returns:
        tuple[float, float | None]: A tuple containing the MLE and variance
            of beta. The variance is `None` if `estimate_var` is `False`.
    """
    if sum(np.array(toxs) == 1) == 0 or sum(np.array(toxs) == 0) == 0:  # type: ignore
        logging.warning(
            "Need heterogeneity in toxic events (toxicity both observed and not) for MLE to exist."
        )
        return np.nan, None

    f = lambda beta: -1 * _compound_toxicity_likelihood(
        F, intercept, beta[0], codified_doses_given, toxs, log=True
    )
    res = minimize(f, x0=np.array([0.0]), method="BFGS")
    var = None
    if res.success:
        beta_hat = res.x[0]
        if estimate_var:
            # Numerical Hessian with step-size control for observed information
            eps = np.finfo(float).eps
            h = (eps ** (1 / 4)) * max(abs(beta_hat), 1.0)
            f_mid = res.fun
            f_plus = f(np.array([beta_hat + h]))  # type: ignore
            f_minus = f(np.array([beta_hat - h]))  # type: ignore
            hessian = (f_plus - 2 * f_mid + f_minus) / (h**2)

            if hessian > 0:
                var = 1.0 / hessian
            else:
                logging.warning(
                    "Numerical Hessian is not positive; falling back to BFGS estimate."
                )
                var = res.hess_inv[0, 0]
        return beta_hat, var
    else:
        logging.warning("Minimization failed; cannot estimate beta_hat.")
        return np.nan, None


def _get_beta_hat_mle_bootstrap(F: Any, intercept: Any, beta_hat: Any, codified_doses_given: Any, B: Any = 200) -> Any:
    """Estimates the variance of the beta MLE using parametric bootstrap.

    Args:
        F (callable): The link function.
        intercept (float): The intercept parameter.
        beta_hat (float): The MLE of beta.
        codified_doses_given (list[float]): The codified dose levels given.
        B (int, optional): The number of bootstrap samples. Defaults to 200.

    Returns:
        float: The estimated variance of beta_hat. Returns `np.nan` if no
            valid bootstrap samples are obtained.
    """
    beta_hats_boot = []
    for _ in range(B):
        tox_probs = [F(dose, intercept, beta_hat) for dose in codified_doses_given]
        toxs_boot = [np.random.binomial(1, p) for p in tox_probs]

        beta_hat_boot, _ = _get_beta_hat_mle(
            F, intercept, codified_doses_given, toxs_boot, estimate_var=False
        )
        if not np.isnan(beta_hat_boot):
            beta_hats_boot.append(beta_hat_boot)

    if len(beta_hats_boot) == 0:
        logging.warning("No valid bootstrap samples obtained.")
        return np.nan

    if len(beta_hats_boot) < B / 2:
        logging.warning("More than half of bootstrap samples failed to produce an MLE.")

    return np.var(beta_hats_boot)


def _estimate_prob_tox_from_param(F: Any, intercept: Any, beta_hat: Any, dose_labels: Any) -> Any:
    """Estimates the probability of toxicity by plugging in a beta estimate.

    Args:
        F (callable): The link function.
        intercept (float): The intercept parameter.
        beta_hat (float): The estimate for beta.
        dose_labels (list[float]): The dose labels for which to estimate
            the probability of toxicity.

    Returns:
        list[float]: A list of estimated probabilities of toxicity.
    """
    post_tox = [F(x, a0=intercept, beta=beta_hat) for x in dose_labels]
    return post_tox


def _get_post_tox_bayes(F: Any, intercept: Any, dose_labels: Any, codified_doses_given: Any, toxs: Any, beta_pdf: Any, use_quick_integration: Any = False) -> Any:
    """Calculates the posterior probability of toxicity using Bayesian integration.

    Args:
        F (callable): The link function.
        intercept (float): The intercept parameter.
        dose_labels (list[float]): The dose labels for which to estimate
            the probability of toxicity.
        codified_doses_given (list[float]): The codified dose levels given.
        toxs (list[int]): The observed toxicity events.
        beta_pdf (callable): The PDF of the prior distribution for beta.
        use_quick_integration (bool, optional): Ignored. Included for backward compatibility.

    Returns:
        list[float]: A list of posterior probabilities of toxicity.
    """
    from clintrials.core.numerics import integrate_posterior_1d

    def logpost(t: Any) -> Any:
        ll = _compound_toxicity_likelihood(
            F, intercept, t, codified_doses_given, toxs, log=True
        )
        return ll + np.log(beta_pdf(t) + 1e-300)

    post_tox = []
    for x in dose_labels:
        prob = integrate_posterior_1d(  # type: ignore
            logpost, lambda t: F(x, a0=intercept, beta=t), _min_beta, _max_beta
        )
        post_tox.append(prob)

    return post_tox


def crm(prior: Any, target: Any, toxicities: Any, dose_levels: Any, intercept: Any = 3, F_func: Any = logistic, inverse_F: Any = inverse_logistic, beta_dist: Any = norm(loc=0, scale=np.sqrt(1.34)), method: Any = "bayes", use_quick_integration: Any = False, estimate_var: Any = False, plugin_mean: Any = True, mle_var_method: Any = "hessian", bootstrap_samples: Any = 200) -> Any:
    """Performs a Continual Reassessment Method (CRM) calculation.

    Args:
        prior (list[float]): A list of prior probabilities of toxicity for
            each dose level.
        target (float): The target toxicity rate.
        toxicities (list[int]): A list of observed toxicity events (1 for
            toxicity, 0 for no toxicity).
        dose_levels (list[int]): A list of the 1-based dose levels given to
            patients.
        intercept (float, optional): The intercept parameter, used with the
            logistic method. Defaults to 3.
        F_func (callable, optional): The link function to use. Defaults to
            `clintrials.core.math.logistic`.
        inverse_F (callable, optional): The inverse link function. Defaults
            to `clintrials.core.math.inverse_logistic`.
        beta_dist (scipy.stats.rv_continuous, optional): The prior
            distribution for the beta parameter. Defaults to a normal
            distribution.
        method (str, optional): The estimation method, either "bayes" or
            "mle". Defaults to "bayes".
        use_quick_integration (bool, optional): Ignored. Included for backward compatibility.
        estimate_var (bool, optional): If `True`, estimates the posterior
            variance of beta. Defaults to `False`.
        plugin_mean (bool, optional): If `True`, plugs the beta estimate into
            the link function. Defaults to `True`.
        mle_var_method (str, optional): The method for estimating the MLE
            variance, either "hessian" or "bootstrap". Defaults to "hessian".
        bootstrap_samples (int, optional): The number of bootstrap samples to
            use if `mle_var_method` is "bootstrap". Defaults to 200.

    Returns:
        tuple: A tuple containing the recommended dose index, the beta
            estimate, the beta variance, and the posterior probabilities of
            toxicity. If `estimate_var` is `True`, the beta standard error
            is also appended as a fifth element.
    """
    from clintrials.validation import validate_matching_lengths

    validate_matching_lengths(toxicities=toxicities, dose_levels=dose_levels)  # type: ignore

    if "logit1" in F_func.__name__ and isinstance(beta_dist, type(norm())):
        alpha0 = np.exp(beta_dist.mean() + beta_dist.var() / 2)
        beta0 = np.log(alpha0)
    else:
        beta0 = beta_dist.mean()

    codified_doses = [
        inverse_F(prior[dl - 1], a0=intercept, beta=beta0) for dl in dose_levels
    ]
    dose_labels = [inverse_F(p, a0=intercept, beta=beta0) for p in prior]
    if method == "bayes":
        beta_hat, var = _get_beta_hat_bayes(
            F_func,
            intercept,
            codified_doses,
            toxicities,
            beta_dist.pdf,
            use_quick_integration,
            estimate_var,
        )
        if plugin_mean:
            post_tox = _estimate_prob_tox_from_param(
                F_func, intercept, beta_hat, dose_labels
            )
        else:
            post_tox = _get_post_tox_bayes(
                F_func,
                intercept,
                dose_labels,
                codified_doses,
                toxicities,
                beta_dist.pdf,
                use_quick_integration,
            )
    elif method == "mle":
        beta_hat, var = _get_beta_hat_mle(
            F_func,
            intercept,
            codified_doses,
            toxicities,
            estimate_var=(mle_var_method == "hessian"),
        )
        if estimate_var and mle_var_method == "bootstrap":
            var = _get_beta_hat_mle_bootstrap(
                F_func, intercept, beta_hat, codified_doses, B=bootstrap_samples
            )

        post_tox = _estimate_prob_tox_from_param(
            F_func, intercept, beta_hat, dose_labels
        )
    else:
        msg = "Only 'bayes' and 'mle' methods are implemented."
        raise ValueError(msg)

    abs_distance_from_target = [abs(x - target) for x in post_tox]
    dose = np.argmin(abs_distance_from_target) + 1

    if estimate_var:
        se = np.sqrt(var) if var is not None else None
        return dose, beta_hat, var, post_tox, se
    else:
        return dose, beta_hat, var, post_tox


class CRM(DoseFindingTrial):
    """An object-oriented implementation of the Continual Reassessment Method (CRM).

    Examples:
        >>> prior_tox_probs = [0.025, 0.05, 0.1, 0.25]
        >>> tox_target = 0.35
        >>> first_dose = 3
        >>> trial_size = 30
        >>> trial = CRM(prior_tox_probs, tox_target, first_dose, trial_size)
        >>> trial.next_dose()
        3
        >>> trial.update([(3,0), (3,0), (3,0)])
        4
        >>> trial.size(), trial.max_size()
        (3, 30)
        >>> trial.update([(4,0), (4,1), (4,1)])
        4
        >>> trial.update([(4,0), (4,1), (4,1)])
        3
        >>> trial.has_more()
        True
    """

    @classmethod
    def get_summary_functions(cls) -> Any:
        """Get summary functions for the CRM protocol."""
        import pandas as pd

        return {
            "N": lambda s, p: len(s),
            "recommended_dose_prob": lambda s, p: pd.Series(
                [x.get("RecommendedDose") for x in s]
            )
            .value_counts(normalize=True)
            .sort_index(),
        }

    def __init__(self, prior: Any, target: Any, first_dose: Any, max_size: Any, F_func: Any = empiric, inverse_F: Any = inverse_empiric, beta_prior: Any = norm(0, np.sqrt(1.34)), method: Any = "bayes", use_quick_integration: Any = False, estimate_var: Any = True, avoid_skipping_untried_escalation: Any = False, avoid_skipping_untried_deescalation: Any = False, lowest_dose_too_toxic_hurdle: Any = None, lowest_dose_too_toxic_certainty: Any = None, coherency_threshold: Any = None, principle_escalation_func: Any = None, termination_func: Any = None, plugin_mean: Any = True, intercept: Any = 3, mle_var_method: Any = "hessian", bootstrap_samples: Any = None) -> None:
        """Initializes a CRM trial object.

        Args:
            prior (list[float]): A list of prior probabilities of toxicity for
                each dose level.
            target (float): The target toxicity rate.
            first_dose (int): The starting dose level (1-based).
            max_size (int): The maximum number of patients in the trial.
            F_func (callable, optional): The link function to use.
                Defaults to `clintrials.core.math.empiric`.
            inverse_F (callable, optional): The inverse link function.
                Defaults to `clintrials.core.math.inverse_empiric`.
            beta_prior (scipy.stats.rv_continuous, optional): The prior
                distribution for the beta parameter. Defaults to a normal
                distribution.
            method (str, optional): The estimation method, either "bayes" or
                "mle". Defaults to "bayes".
            use_quick_integration (bool, optional): Ignored. Included for backward compatibility.
            estimate_var (bool, optional): If `True`, estimates the posterior
                variance of beta. Defaults to `True`.
            avoid_skipping_untried_escalation (bool, optional): If `True`,
                avoids skipping untried doses when escalating. Defaults to
                `False`.
            avoid_skipping_untried_deescalation (bool, optional): If `True`,
                avoids skipping untried doses when de-escalating. Defaults to
                `False`.
            lowest_dose_too_toxic_hurdle (float, optional): The toxicity
                hurdle for the lowest dose. If the posterior probability that
                the toxicity of the lowest dose exceeds this hurdle is
                greater than `lowest_dose_too_toxic_certainty`, the trial is
                stopped. Both must be positive for the test to be invoked.
                Defaults to 0.0.
            lowest_dose_too_toxic_certainty (float, optional): The certainty
                for the lowest dose toxicity hurdle. See
                `lowest_dose_too_toxic_hurdle`. Defaults to 0.0.
            coherency_threshold (float, optional): If positive, prevents
                escalation if the observed toxicity rate at the current dose
                exceeds this value. Defaults to 0.0.
            principle_escalation_func (callable, optional): An optional
                function that takes the trial cases and returns the next dose
                to be given, or `None` to use the CRM method. This allows
                for custom escalation strategies. Defaults to `None`.
            termination_func (callable, optional): An optional function that
                takes the trial instance and returns `True` if the trial
                should terminate. Defaults to `None`.
            plugin_mean (bool, optional): If `True`, plugs the beta estimate
                into the link function. Defaults to `True`.
            intercept (float, optional): The intercept parameter, used with
                the logistic method. Defaults to 3.
            mle_var_method (str, optional): The method for estimating the
                MLE variance, either "hessian" or "bootstrap". Defaults to
                "hessian".
            bootstrap_samples (int, optional): The number of bootstrap
                samples to use if `mle_var_method` is "bootstrap". Defaults
                to 200.
        """
        from clintrials.core.schema import CRMSchema

        # Build kwargs for schema defaults
        schema_kwargs = {
            "prior": prior,
            "target": target,
            "first_dose": first_dose,
            "max_size": max_size,
        }
        if lowest_dose_too_toxic_hurdle is not None:
            schema_kwargs["lowest_dose_too_toxic_hurdle"] = lowest_dose_too_toxic_hurdle
        if lowest_dose_too_toxic_certainty is not None:
            schema_kwargs["lowest_dose_too_toxic_certainty"] = (
                lowest_dose_too_toxic_certainty
            )
        if coherency_threshold is not None:
            schema_kwargs["coherency_threshold"] = coherency_threshold
        if bootstrap_samples is not None:
            schema_kwargs["bootstrap_samples"] = bootstrap_samples

        config = CRMSchema(**schema_kwargs)

        DoseFindingTrial.__init__(
            self,
            first_dose=config.first_dose,
            num_doses=len(config.prior),
            max_size=config.max_size,
        )

        self.prior = config.prior
        self.target = config.target
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.beta_prior = beta_prior
        self.method = method
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation
        self.lowest_dose_too_toxic_hurdle = config.lowest_dose_too_toxic_hurdle
        self.lowest_dose_too_toxic_certainty = config.lowest_dose_too_toxic_certainty
        self.coherency_threshold = config.coherency_threshold
        self.principle_escalation_func = principle_escalation_func
        self.termination_func = termination_func
        self.plugin_mean = plugin_mean
        self.intercept = intercept
        self.mle_var_method = mle_var_method
        self.bootstrap_samples = config.bootstrap_samples

        if lowest_dose_too_toxic_hurdle and lowest_dose_too_toxic_certainty:
            if not self.estimate_var:
                logging.warning(
                    "To monitor toxicity at the lowest dose, beta variance estimation was enabled."
                )
            self.estimate_var = True
        self.beta_hat, self.beta_var = beta_prior.mean(), beta_prior.var()
        self.beta_se = np.sqrt(self.beta_var) if self.beta_var is not None else None
        self.post_tox = list(self.prior)

    def _DoseFindingTrial__reset(self) -> Any:
        self.beta_hat, self.beta_var = self.beta_prior.mean(), self.beta_prior.var()
        self.beta_se = np.sqrt(self.beta_var) if self.beta_var is not None else None
        self.post_tox = self.prior

    def _DoseFindingTrial__calculate_next_dose(self) -> Any:
        if self.principle_escalation_func:
            cases = zip(self._doses, self._toxicities)
            proposed_dose = self.principle_escalation_func(cases)
            if proposed_dose is not None:
                return proposed_dose

        current_dose = self.next_dose()
        max_dose_given = self.maximum_dose_given()
        min_dose_given = self.minimum_dose_given()
        crm_res = crm(
            prior=self.prior,
            target=self.target,
            toxicities=self._toxicities,
            dose_levels=self._doses,
            intercept=self.intercept,
            F_func=self.F_func,
            inverse_F=self.inverse_F,
            beta_dist=self.beta_prior,
            method=self.method,
            use_quick_integration=self.use_quick_integration,
            estimate_var=self.estimate_var,
            plugin_mean=self.plugin_mean,
            mle_var_method=self.mle_var_method,
            bootstrap_samples=self.bootstrap_samples,
        )
        proposed_dose = crm_res[0]
        self.beta_hat = crm_res[1]
        self.beta_var = crm_res[2]
        self.post_tox = crm_res[3]
        if self.estimate_var:
            self.beta_se = crm_res[4]
        else:
            self.beta_se = None

        if self.lowest_dose_too_toxic_hurdle and self.lowest_dose_too_toxic_certainty:
            labels = [
                self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean())
                for p in self.prior
            ]
            beta_sample = norm(loc=beta_hat, scale=np.sqrt(beta_var)).rvs(1000000)  # type: ignore
            p0_sample = self.F_func(labels[0], a0=self.intercept, beta=beta_sample)
            p0_tox = np.mean(p0_sample > self.lowest_dose_too_toxic_hurdle)

            if p0_tox > self.lowest_dose_too_toxic_certainty:
                proposed_dose = 0
                self._status = -1
                return proposed_dose

        if self.coherency_threshold and proposed_dose > current_dose:
            tox_rate_at_current = self.observed_toxicity_rates()[current_dose - 1]  # type: ignore
            if (
                not np.isnan(tox_rate_at_current)
                and tox_rate_at_current > self.coherency_threshold
            ):
                proposed_dose = current_dose
                return proposed_dose

        if (
            self.avoid_skipping_untried_escalation
            and max_dose_given
            and proposed_dose - max_dose_given > 1
        ):
            proposed_dose = max_dose_given + 1
            return proposed_dose
        elif (
            self.avoid_skipping_untried_deescalation
            and min_dose_given
            and min_dose_given - proposed_dose > 1
        ):
            proposed_dose = min_dose_given - 1
            return proposed_dose

        return proposed_dose

    def prob_tox(self) -> npt.NDArray[np.float64]:  # type: ignore
        """Gets the posterior probabilities of toxicity for each dose level.

        Returns:
            list[float]: A list of posterior probabilities of toxicity.
        """
        return list(self.post_tox)

    def _prob_tox_exceeds_quadrature(self, tox_cutoff: Any, deg: Any = REGISTRY["crm_deg"]) -> Any:
        """Posterior Pr(toxicity > cutoff) using Gauss--Hermite quadrature."""
        mu0 = self.beta_prior.mean()
        sd0 = np.sqrt(self.beta_prior.var())
        nodes, weights = np.polynomial.hermite.hermgauss(deg)  # type: ignore
        betas = mu0 + np.sqrt(2) * sd0 * nodes  # type: ignore
        log_w = np.log(weights)  # type: ignore
        dose_labels = [
            self.inverse_F(self.prior[d - 1], a0=self.intercept, beta=mu0)
            for d in self.doses()
        ]
        ll = _compound_toxicity_likelihood(
            self.F_func,
            self.intercept,
            betas,
            dose_labels,
            self.toxicities(),
            log=True,
        )
        log_post = log_w + ll
        log_denom = logsumexp(log_post)
        post_weights = np.exp(log_post - log_denom)
        labels = [
            self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean())
            for p in self.prior
        ]
        out = []
        for lab in labels:
            tox_probs = self.F_func(lab, a0=self.intercept, beta=betas)
            out.append(np.sum(post_weights * (tox_probs > tox_cutoff)))
        return np.array(out)

    def prob_tox_exceeds(self, tox_cutoff: Any, backend: Any = "quadrature", n: Any = 10**6) -> Any:
        """Calculates the posterior probability that toxicity exceeds a cutoff.

        Args:
            tox_cutoff (float): The toxicity cutoff.
            backend (str, optional): The calculation backend, either
                "quadrature" or "bootstrap". Defaults to "quadrature".
            n (int, optional): The number of samples for the "bootstrap"
                backend. Defaults to 10**6.

        Returns:
            numpy.ndarray: An array of posterior probabilities for each
                dose level.
        """
        if backend == "quadrature":
            return self._prob_tox_exceeds_quadrature(tox_cutoff)
        if backend == "bootstrap":
            warnings.warn(
                "bootstrap backend is deprecated", DeprecationWarning, stacklevel=2
            )
            if self.estimate_var:
                labels = [
                    self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean())
                    for p in self.prior
                ]
                beta_sample = norm(loc=self.beta_hat, scale=np.sqrt(self.beta_var)).rvs(
                    n
                )
                p0_sample = [
                    self.F_func(label, a0=self.intercept, beta=beta_sample)
                    for label in labels
                ]
                return np.array([np.mean(x > tox_cutoff) for x in p0_sample])
            raise Exception(
                "CRM can only estimate posterior probabilities when estimate_var=True"
            )
        raise ValueError("Unknown backend")

    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        if not DoseFindingTrial.has_more(self):
            return False
        if self.termination_func:
            return not self.termination_func(self)
        else:
            return True

    def optimal_decision(self, prob_tox: Sequence[float]) -> int:
        """Gets the optimal dose choice for a given dose-toxicity curve.

        Args:
            prob_tox (list[float]): A list of toxicity probabilities for each
                dose level.

        Returns:
            int: The optimal 1-based dose level.
        """
        return np.argmin(np.abs(prob_tox - self.target)) + 1  # type: ignore

    def get_tox_prob_quantile(self, p: Any) -> Any:
        """Gets the quantiles of the toxicity probabilities for each dose.

        This method uses a normal approximation.

        Args:
            p (float): The quantile to calculate (e.g., 0.05 for the 5th
                percentile).

        Returns:
            list[float]: A list of toxicity probability quantiles for each
                dose level.
        """
        norm_crit = norm.ppf(p)
        beta_est = self.beta_hat - norm_crit * np.sqrt(self.beta_var)
        labels = [
            self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean())
            for p in self.prior
        ]
        p = [self.F_func(x, a0=self.intercept, beta=beta_est) for x in labels]
        return p

    def plot_toxicity_probabilities(self, chart_title: Optional[str] = None, use_ggplot: bool = False) -> Any:
        """Plots the prior and posterior dose-toxicity curves.

        Args:
            chart_title (str, optional): The title for the chart.
                Defaults to a descriptive title.
            use_ggplot (bool, optional): Ignored. Included for backwards compatibility.

        Returns:
            A plot object.
        """
        from clintrials.core.viz_interface import get_visualization_provider

        viz = get_visualization_provider()  # type: ignore

        return viz.plot_crm_toxicity_probabilities(self, chart_title=chart_title)


def crm_dtp_detail(trial: Any) -> Any:
    """Gets CRM-specific details for DTP reporting.

    Args:
        trial (CRM): An instance of the CRM class.

    Returns:
        collections.OrderedDict: A dictionary with CRM-specific details.
    """
    to_return = OrderedDict()

    if trial.beta_hat is not None:
        to_return["BetaHat"] = atomic_to_json(trial.beta_hat)
    if trial.beta_var is not None:
        to_return["BetaVar"] = atomic_to_json(trial.beta_var)
    if hasattr(trial, "beta_se") and trial.beta_se is not None:
        to_return["BetaSE"] = atomic_to_json(trial.beta_se)

    if trial.prob_tox() is not None:
        to_return["ProbTox"] = iterable_to_json(trial.prob_tox())
        for i, dl in enumerate(trial.dose_levels()):
            to_return[f"ProbTox{dl}"] = trial.prob_tox()[i]

    return to_return


__all__ = ["CRM", "crm", "crm_dtp_detail"]


# Inject module-level docstring
if __doc__:
    __doc__ = __doc__.format(**REGISTRY)
