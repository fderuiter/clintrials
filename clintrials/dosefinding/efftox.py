from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

"""
An implementation of Thall & Cook's EffTox design for dose-finding in
clinical trials.


Random Seed Strategy: {efftox_seed_strategy}
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import logging
from collections import OrderedDict

from scipy.optimize import brentq

from clintrials.core.math import inverse_logit, logit
from clintrials.core.stats import norm  # type: ignore
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial
from clintrials.utils import atomic_to_json, iterable_to_json


def scale_doses(real_doses: Any) -> Any:
    """Scales the doses for use in the EffTox model.

    Args:
        real_doses (list[float]): The actual dose amounts.

    Returns:
        numpy.ndarray: The scaled doses.
    """
    return np.log(real_doses) - np.mean(np.log(real_doses))


def efftox_priors_from_skeleton(real_doses: Any, prior_tox_probs: Any, prior_eff_probs: Any) -> Any:
    """Elicits principled EffTox priors from a dose-response skeleton.

    This function fits the EffTox link functions to the provided prior
    probabilities using least squares on the logit scale.

    Args:
        real_doses (list[float]): The actual dose amounts.
        prior_tox_probs (list[float]): Prior toxicity probabilities at each dose.
        prior_eff_probs (list[float]): Prior efficacy probabilities at each dose.

    Returns:
        list[scipy.stats.norm]: A list of 6 normal distributions for the
            parameters (mu_T, beta_T, mu_E, beta1_E, beta2_E, psi).
    """
    scaled_x = scale_doses(real_doses)

    # Toxicity: logit(pi_T) = mu_T + beta_T * x
    logit_tox = logit(np.array(prior_tox_probs))
    beta_T_mean, mu_T_mean = np.polyfit(scaled_x, logit_tox, 1)  # type: ignore[misc]

    # Efficacy: logit(pi_E) = mu_E + beta1_E * x + beta2_E * x^2
    logit_eff = logit(np.array(prior_eff_probs))
    beta2_E_mean, beta1_E_mean, mu_E_mean = np.polyfit(scaled_x, logit_eff, 2)  # type: ignore[misc]

    # Principled default SDs:
    # Intercepts and linear slopes: 2.0
    # Quadratic term: 0.2 (usually smaller as it's for curvature)
    # Association parameter psi: mean 0, SD 1.0
    priors = [
        norm(loc=mu_T_mean, scale=2.0),  # type: ignore[has-type]
        norm(loc=beta_T_mean, scale=2.0),  # type: ignore[has-type]
        norm(loc=mu_E_mean, scale=2.0),  # type: ignore[has-type]
        norm(loc=beta1_E_mean, scale=2.0),  # type: ignore[has-type]
        norm(loc=beta2_E_mean, scale=0.2),  # type: ignore[has-type]
        norm(loc=0.0, scale=1.0),
    ]
    return priors


def validate_efftox_priors(priors: Any, scaled_doses: Any) -> Any:
    """Validates that the EffTox priors imply sensible dose-response shapes.

    Args:
        priors (list): A list of 6 prior distributions.
        scaled_doses (list[float]): The scaled dose levels.

    Raises:
        ValueError: If the priors imply non-sensible dose-response shapes.
    """
    from clintrials.validation import validate_expected_length

    validate_expected_length(priors, 6, "priors")

    beta_T = priors[1].mean()

    # Check if toxicity is non-decreasing
    if beta_T < 0:
        raise ValueError("Toxicity prior slope (beta_T) should be non-negative.")


def _eta_T(scaled_dose: Any, mu: Any, beta: Any) -> Any:
    """Calculates the linear predictor for toxicity.

    Args:
        scaled_dose (float): The scaled dose.
        mu (float): The intercept parameter.
        beta (float): The slope parameter.

    Returns:
        float: The linear predictor for toxicity.
    """
    return mu + beta * scaled_dose


def _eta_E(scaled_dose: Any, mu: Any, beta1: Any, beta2: Any) -> Any:
    """Calculates the linear predictor for efficacy.

    Args:
        scaled_dose (float): The scaled dose.
        mu (float): The intercept parameter.
        beta1 (float): The linear slope parameter.
        beta2 (float): The quadratic slope parameter.

    Returns:
        float: The linear predictor for efficacy.
    """
    return mu + beta1 * scaled_dose + beta2 * scaled_dose**2


def _pi_T(scaled_dose: Any, mu: Any, beta: Any) -> Any:
    """Calculates the probability of toxicity.

    Args:
        scaled_dose (float): The scaled dose.
        mu (float): The intercept parameter.
        beta (float): The slope parameter.

    Returns:
        float: The probability of toxicity.
    """
    return inverse_logit(_eta_T(scaled_dose, mu, beta))


def _pi_E(scaled_dose: Any, mu: Any, beta1: Any, beta2: Any) -> Any:
    """Calculates the probability of efficacy.

    Args:
        scaled_dose (float): The scaled dose.
        mu (float): The intercept parameter.
        beta1 (float): The linear slope parameter.
        beta2 (float): The quadratic slope parameter.

    Returns:
        float: The probability of efficacy.
    """
    return inverse_logit(_eta_E(scaled_dose, mu, beta1, beta2))


def _pi_ab(scaled_dose: Any, tox: Any, eff: Any, mu_T: Any, beta_T: Any, mu_E: Any, beta1_E: Any, beta2_E: Any, psi: Any) -> Any:
    """Calculates the likelihood of a joint toxicity-efficacy outcome.

    Args:
        scaled_dose (float): The scaled dose.
        tox (int): The toxicity outcome (1 or 0).
        eff (int): The efficacy outcome (1 or 0).
        mu_T (float): The intercept for toxicity.
        beta_T (float): The slope for toxicity.
        mu_E (float): The intercept for efficacy.
        beta1_E (float): The linear slope for efficacy.
        beta2_E (float): The quadratic slope for efficacy.
        psi (float): The association parameter.

    Returns:
        float: The likelihood of the outcome.
    """
    from clintrials.core.math import fgm_joint_prob

    p_E = _pi_E(scaled_dose, mu_E, beta1_E, beta2_E)
    p_T = _pi_T(scaled_dose, mu_T, beta_T)
    return fgm_joint_prob(eff, tox, p_E, p_T, psi)


def _L_n(D: Any, mu_T: Any, beta_T: Any, mu_E: Any, beta1_E: Any, beta2_E: Any, psi: Any) -> Any:
    """Calculates the compound likelihood for a set of observations.

    Args:
        D (list[tuple[float, int, int]]): A list of observations, where
            each observation is a tuple of (scaled_dose, toxicity,
            efficacy).
        mu_T (float): The intercept for toxicity.
        beta_T (float): The slope for toxicity.
        mu_E (float): The intercept for efficacy.
        beta1_E (float): The linear slope for efficacy.
        beta2_E (float): The quadratic slope for efficacy.
        psi (float): The association parameter.

    Returns:
        float: The compound likelihood.
    """
    response = np.ones(len(mu_T))
    for scaled_dose, tox, eff in D:
        p = _pi_ab(scaled_dose, tox, eff, mu_T, beta_T, mu_E, beta1_E, beta2_E, psi)
        response *= p
    return response


def _get_posterior_sample(cases: Any, priors: Any, rng: Any = None, n: Any = 10**5, epsilon: Any = 1e-6, k_sd: Any = 6.0, max_iter: Any = 3, mass_threshold: Any = 0.999999) -> Any:
    """Generates a posterior sample with adaptive integration limits.

    Args:
        cases (list[tuple]): A list of cases (scaled_dose, tox, eff).
        priors (list): A list of 6 prior distributions.
        rng (numpy.random.Generator, optional): Random number generator.
        n (int, optional): Number of points for Monte Carlo integration.
            Defaults to 10**5.
        epsilon (float, optional): Initial quantile for limits. Defaults to
            1e-6.
        k_sd (float, optional): Number of standard deviations for limit
            coverage. Defaults to 6.0.
        max_iter (int, optional): Maximum number of refinement iterations.
            Defaults to 3.
        mass_threshold (float, optional): Threshold for Gaussian-approximate
            mass coverage within boundaries. Defaults to 0.999999.

    Returns:
        ProbabilityDensitySample: The posterior sample object.
    """
    if rng is None:
        from clintrials.core.rng import get_rng
        rng = get_rng()  # type: ignore

    limits = [(dist.ppf(epsilon), dist.ppf(1 - epsilon)) for dist in priors]

    lik_integrand = (
        lambda x: _L_n(cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])
        * priors[0].pdf(x[:, 0])
        * priors[1].pdf(x[:, 1])
        * priors[2].pdf(x[:, 2])
        * priors[3].pdf(x[:, 3])
        * priors[4].pdf(x[:, 4])
        * priors[5].pdf(x[:, 5])
    )

    from clintrials.core.numerics import adaptive_mc_integration

    refined_limits, pds = adaptive_mc_integration(  # type: ignore
        lik_integrand,
        limits,
        rng,
        n=n,
        max_iter=max_iter,
        mass_threshold=mass_threshold,
        k_sd=k_sd,
    )

    return pds


def efftox_get_posterior_probs(cases: Any, priors: Any, scaled_doses: Any, tox_cutoff: Any, eff_cutoff: Any, n: Any = 10**5, epsilon: Any = 1e-6, rng: Any = None, **kwargs: Any) -> Any:
    """Calculates posterior probabilities for an EffTox trial.

    This function uses Monte Carlo integration to evaluate the posterior
    integrals.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases, where each case
            is a tuple of (dose, toxicity, efficacy).
        priors (list): A list of 6 prior distributions for the model
            parameters.
        scaled_doses (list[float]): The scaled dose levels.
        tox_cutoff (float): The maximum acceptable toxicity probability.
        eff_cutoff (float): The minimum acceptable efficacy probability.
        n (int, optional): The number of points for Monte Carlo integration.
            Defaults to 10**5.
        epsilon (float, optional): A small number to define the integration
            range. Defaults to 1e-6.
        rng (numpy.random.Generator, optional): Random number generator.
        **kwargs: Additional arguments for limit refinement.

    Returns:
        tuple[list, ProbabilityDensitySample]: A tuple containing a list of
            posterior probabilities for each dose, and the
            `ProbabilityDensitySample` object.
    """
    from clintrials.validation import validate_expected_length

    validate_expected_length(priors, 6, "priors")

    # Convert dose-levels given to dose amounts given
    if len(cases) > 0:
        dose_levels, tox_events, eff_events = zip(*cases)
        scaled_doses_given = [scaled_doses[x - 1] for x in dose_levels]
        _cases = list(zip(scaled_doses_given, tox_events, eff_events))
    else:
        _cases = []

    limit_args = {
        k: v for k, v in kwargs.items() if k in ["k_sd", "max_iter", "mass_threshold"]
    }
    pds = _get_posterior_sample(_cases, priors, rng=rng, n=n, epsilon=epsilon, **limit_args)
    samp = pds._samp

    probs = []
    for x in scaled_doses:
        tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
        eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
        probs.append(
            (
                pds.expectation(tox_probs),
                pds.expectation(eff_probs),
                pds.expectation(tox_probs < tox_cutoff),
                pds.expectation(eff_probs > eff_cutoff),
            )
        )

    return probs, pds


def efftox_get_posterior_params(cases: Any, priors: Any, scaled_doses: Any, n: Any = 10**5, epsilon: Any = 1e-6, rng: Any = None, **kwargs: Any) -> Any:
    """Calculates posterior parameter estimates for an EffTox trial.

    This function uses Monte Carlo integration to evaluate the posterior
    integrals.

    Args:
        cases (list[tuple[int, int, int]]): A list of cases.
        priors (list): A list of 6 prior distributions.
        scaled_doses (list[float]): The scaled dose levels.
        n (int, optional): The number of points for Monte Carlo integration.
            Defaults to 10**5.
        epsilon (float, optional): A small number to define the integration
            range. Defaults to 1e-6.
        rng (numpy.random.Generator, optional): Random number generator.
        **kwargs: Additional arguments for limit refinement.

    Returns:
        tuple[list, ProbabilityDensitySample]: A tuple containing a list of
            posterior parameter estimates and the `ProbabilityDensitySample`
            object.
    """
    from clintrials.validation import validate_expected_length

    validate_expected_length(priors, 6, "priors")

    # Convert dose-levels given to dose amounts given
    if len(cases) > 0:
        dose_levels, tox_events, eff_events = zip(*cases)
        scaled_doses_given = [scaled_doses[x - 1] for x in dose_levels]
        _cases = list(zip(scaled_doses_given, tox_events, eff_events))
    else:
        _cases = []

    limit_args = {
        k: v for k, v in kwargs.items() if k in ["k_sd", "max_iter", "mass_threshold"]
    }
    pds = _get_posterior_sample(_cases, priors, rng=rng, n=n, epsilon=epsilon, **limit_args)
    samp = pds._samp

    params = []
    params.append(
        (
            pds.expectation(samp[:, 0]),
            pds.expectation(samp[:, 1]),
            pds.expectation(samp[:, 2]),
            pds.expectation(samp[:, 3]),
            pds.expectation(samp[:, 4]),
            pds.expectation(samp[:, 5]),
        )
    )

    return params, pds


class LpNormCurve:
    """Fits an indifference contour using an L-p norm.

    This class fits an indifference contour based on three points:
    - Efficacy when toxicity is impossible.
    - Toxicity when efficacy is guaranteed.
    - An equally desirable hinge point.
    """

    def __init__(self, minimum_tolerable_efficacy: Any, maximum_tolerable_toxicity: Any, hinge_prob_eff: Any, hinge_prob_tox: Any) -> None:
        """Initializes an LpNormCurve object.

        Args:
            minimum_tolerable_efficacy (float): The tolerable efficacy when
                toxicity is impossible.
            maximum_tolerable_toxicity (float): The tolerable toxicity when
                efficacy is guaranteed.
            hinge_prob_eff (float): The probability of efficacy at the hinge
                point.
            hinge_prob_tox (float): The probability of toxicity at the hinge
                point.

        Raises:
            ValueError: If the hinge point probabilities are not within the
                bounds defined by the other parameters.
        """
        if hinge_prob_tox >= maximum_tolerable_toxicity:
            raise ValueError(
                "Probability of toxicity at hinge point should be less than toxicity upper bound."
            )
        if hinge_prob_eff <= minimum_tolerable_efficacy:
            raise ValueError(
                "Probability of efficacy at hinge point should be greater than efficacy lower bound."
            )

        def _find_p(p: Any) -> Any:
            a = (1 - hinge_prob_eff) / (1 - minimum_tolerable_efficacy)
            b = hinge_prob_tox / maximum_tolerable_toxicity
            return a**p + b**p - 1

        self.minimum_tolerable_efficacy = minimum_tolerable_efficacy
        self.maximum_tolerable_toxicity = maximum_tolerable_toxicity
        self.p = brentq(_find_p, 0, 100)
        self.hinge_points = [
            (minimum_tolerable_efficacy, 0),
            (1, maximum_tolerable_toxicity),
            (hinge_prob_eff, hinge_prob_tox),
        ]

    def __call__(self, prob_eff: Any, prob_tox: Any) -> Any:
        """Calculates the utility of a given efficacy-toxicity pair.

        Args:
            prob_eff (float): The probability of efficacy.
            prob_tox (float): The probability of toxicity.

        Returns:
            float: The utility value.
        """
        x = prob_eff
        y = prob_tox
        if np.all(0 < x) and np.all(x < 1) and np.all(0 < y) and np.all(y < 1):
            a = (1 - x) / (1 - self.minimum_tolerable_efficacy)
            b = y / self.maximum_tolerable_toxicity
            r_to_the_p = a**self.p + b**self.p
            return 1 - r_to_the_p ** (1.0 / self.p)
        else:
            response = np.zeros_like(x, dtype=float)
            response *= np.nan
            return response

    def solve(self, delta: Any, *, prob_eff: Any = None, prob_tox: Any = None, bounds: Any = (0, 1), tol: Any = 1e-6, maxiter: Any = 100) -> Any:
        """Solves for one probability given the other and a utility delta.

        Args:
            delta (float): The utility delta.
            prob_eff (float, optional): The probability of efficacy.
            prob_tox (float, optional): The probability of toxicity.
            bounds (tuple, optional): The search bounds for the missing
                probability. Defaults to (0, 1).
            tol (float, optional): The convergence tolerance. Defaults to 1e-6.
            maxiter (int, optional): The maximum number of iterations.
                Defaults to 100.

        Returns:
            float: The solved probability.

        Raises:
            ValueError: If parameters are invalid or the delta is infeasible.
        """
        if (prob_eff is None) == (prob_tox is None):
            raise ValueError("Exactly one of prob_eff or prob_tox must be specified.")

        eps = 1e-9
        low = max(bounds[0], eps)
        high = min(bounds[1], 1 - eps)

        if prob_eff is not None:

            def g(p: Any) -> Any:
                return self.__call__(prob_eff, p) - delta

        else:

            def g(p: Any) -> Any:
                return self.__call__(p, prob_tox) - delta

        try:
            return brentq(g, low, high, xtol=tol, maxiter=maxiter)
        except ValueError as e:
            raise ValueError(
                f"Utility delta {delta} is infeasible for the given probabilities."
            ) from e

    def get_tox(self, eff: Any, util: Any = 0.0) -> Any:
        """Gets the equivalent toxicity probability for a given efficacy and utility.

        Args:
            eff (float): The probability of efficacy.
            util (float, optional): The utility value. Defaults to 0.0.

        Returns:
            float: The equivalent probability of toxicity.
        """
        p = self.p
        eff0 = self.minimum_tolerable_efficacy
        tox1 = self.maximum_tolerable_toxicity
        a = (1 - eff) / (1 - eff0)
        return tox1 * ((1 - util) ** p - a**p) ** (1 / p)



class InverseQuadraticCurve:
    """Fits an indifference contour using an inverse quadratic curve."""

    def __init__(self, points: Any) -> None:
        """Initializes an InverseQuadraticCurve object.

        Args:
            points (list[tuple[float, float]]): A list of (probability of
                efficacy, probability of toxicity) points.

        Raises:
            ValueError: If the points do not fit an ABC curve well.
        """
        x = np.array([z for z, _ in points])
        y = np.array([z for _, z in points])
        z = 1 / x
        import statsmodels.api as sm

        lm = sm.OLS(y, np.column_stack((np.ones_like(z), z, z**2))).fit()
        a, b, c = lm.params
        f = lambda x: a + b / x + c / x**2
        # Check f is not a terrible fit
        if sum(np.abs(f(x) - y)) > 0.00001:  # type: ignore
            raise ValueError("%s do not fit an ABC curve well" % points)
        self.f = f
        self.a, self.b, self.c = a, b, c

    def __call__(self, prob_eff: Any, prob_tox: Any) -> Any:
        """Calculates the utility of a given efficacy-toxicity pair.

        Args:
            prob_eff (float): The probability of efficacy.
            prob_tox (float): The probability of toxicity.

        Returns:
            float: The utility value.
        """
        x = prob_eff
        y = prob_tox
        if 0 < x < 1 and 0 < y < 1:
            gradient = 1.0 * y / (x - 1)

            def intersection_expression(X: Any, m: Any, f: Any) -> Any:
                return m * (X - 1) - f(X)

            try:
                x_00 = brentq(
                    intersection_expression, 0.0001, 1, args=(gradient, self.f)
                )
            except ValueError:
                # If brentq fails, use cubic root finding for robustness.
                m = gradient
                coeffs = [m, -(m + self.a), -self.b, -self.c]
                roots = np.roots(coeffs)
                real_roots = roots[np.isreal(roots)].real  # type: ignore[index]
                valid_roots = real_roots[(real_roots > 0) & (real_roots <= 1.00000001)]
                if len(valid_roots) == 0:
                    return np.nan
                x_00 = min(valid_roots[0], 1.0)

            y_00 = self.f(x_00)  # type: ignore
            d1 = np.sqrt((x_00 - 1) ** 2 + y_00**2)
            d2 = np.sqrt((x - 1) ** 2 + y**2)

            return d1 / d2 - 1
        else:
            return np.nan

    def solve(self, delta: Any, *, prob_eff: Any = None, prob_tox: Any = None, bounds: Any = (0, 1), tol: Any = 1e-6, maxiter: Any = 100) -> Any:
        """Solves for one probability given the other and a utility delta.

        Args:
            delta (float): The utility delta.
            prob_eff (float, optional): The probability of efficacy.
            prob_tox (float, optional): The probability of toxicity.
            bounds (tuple, optional): The search bounds for the missing
                probability. Defaults to (0, 1).
            tol (float, optional): The convergence tolerance. Defaults to 1e-6.
            maxiter (int, optional): The maximum number of iterations.
                Defaults to 100.

        Returns:
            float: The solved probability.

        Raises:
            ValueError: If parameters are invalid or the delta is infeasible.
        """
        if (prob_eff is None) == (prob_tox is None):
            raise ValueError("Exactly one of prob_eff or prob_tox must be specified.")

        eps = 1e-9
        low = max(bounds[0], eps)
        high = min(bounds[1], 1 - eps)

        if prob_eff is not None:

            def g(p: Any) -> Any:
                return self.__call__(prob_eff, p) - delta

        else:

            def g(p: Any) -> Any:
                return self.__call__(p, prob_tox) - delta

        try:
            return brentq(g, low, high, xtol=tol, maxiter=maxiter)
        except ValueError as e:
            raise ValueError(
                f"Utility delta {delta} is infeasible for the given probabilities."
            ) from e





class EffTox(EfficacyToxicityDoseFindingTrial):
    """An object-oriented implementation of Thall & Cook's EffTox trial design.

    See Thall, P.F. & Cook, J.D. (2004) - Dose-Finding Based on
    Efficacy-Toxicity Trade-Offs.
    """

    @classmethod
    def get_summary_functions(cls) -> Any:
        """Get summary functions for the EffTox protocol."""
        return {
            "N": lambda s, p: len(s),
            "recommended_dose_prob": lambda s, p: pd.Series(
                [x.get("RecommendedDose") for x in s]
            )
            .value_counts(normalize=True)
            .sort_index()
            .to_dict(),
            "prob_accept_tox": lambda s, p: pd.Series(
                [x.get(f"ProbAccTox{x.get('RecommendedDose')}", 0) > 0.5 for x in s]
            ).mean(),
            "prob_accept_eff": lambda s, p: pd.Series(
                [x.get(f"ProbAccEff{x.get('RecommendedDose')}", 0) > 0.5 for x in s]
            ).mean(),
        }

    def __init__(self, real_doses: Any, theta_priors: Any = None, tox_cutoff: Any = None, eff_cutoff: Any = None, tox_certainty: Any = None, eff_certainty: Any = None, metric: Any = None, max_size: Any = None, first_dose: Any = None, prior_tox_probs: Any = None, prior_eff_probs: Any = None, avoid_skipping_untried_escalation: Any = True, avoid_skipping_untried_deescalation: Any = True, num_integral_steps: Any = 10**5, epsilon: Any = 1e-6, k_sd: Any = 6.0, max_iter: Any = 3, mass_threshold: Any = 0.999999) -> None:
        """Initializes an EffTox trial object.

        Args:
            real_doses (list[float]): A list of the actual dose amounts.
            theta_priors (list, optional): A list of 6 prior distributions
                for the model parameters. If `None`, principled priors are
                elicited from `prior_tox_probs` and `prior_eff_probs`.
            tox_cutoff (float): The maximum acceptable probability of
                toxicity.
            eff_cutoff (float): The minimum acceptable probability of
                efficacy.
            tox_certainty (float): The posterior certainty required that
                toxicity is less than the cutoff.
            eff_certainty (float): The posterior certainty required that
                efficacy is greater than the cutoff.
            metric (LpNormCurve | InverseQuadraticCurve): An object for
                calculating the utility of efficacy-toxicity pairs.
            max_size (int): The maximum number of patients in the trial.
                Note: In EffTox, `max_size` appears before `first_dose` in the
                initialization signature, differing from other design patterns.
            first_dose (int, optional): The starting dose level (1-based).
                Note: This parameter follows `max_size` in EffTox, differing
                from typical patterns. Defaults to 1.
            prior_tox_probs (list[float], optional): Prior toxicity
                probabilities at each dose level. Used for elicitation if
                `theta_priors` is `None`.
            prior_eff_probs (list[float], optional): Prior efficacy
                probabilities at each dose level. Used for elicitation if
                `theta_priors` is `None`.
            avoid_skipping_untried_escalation (bool, optional): If `True`,
                avoids skipping untried doses when escalating. Defaults to
                `True`.
            avoid_skipping_untried_deescalation (bool, optional): If `True`,
                avoids skipping untried doses when de-escalating. Defaults
                to `True`.
            num_integral_steps (int, optional): The number of points for
                Monte Carlo integration. Defaults to 10**5.
            epsilon (float, optional): A small number to define the
                integration range. Defaults to 1e-6.
            k_sd (float, optional): Number of standard deviations for limit
                coverage. Defaults to 6.0.
            max_iter (int, optional): Maximum number of refinement iterations.
                Defaults to 3.
            mass_threshold (float, optional): Threshold for Gaussian-approximate
                mass coverage within boundaries. Defaults to 0.999999.

        Raises:
            ValueError: If required parameters are missing or if priors are
                invalid.
        """
        from clintrials.core.schema import EffToxSchema

        # Build schema kwargs
        schema_kwargs = {"real_doses": real_doses}
        if prior_tox_probs is not None:
            schema_kwargs["prior_tox_probs"] = prior_tox_probs
        if prior_eff_probs is not None:
            schema_kwargs["prior_eff_probs"] = prior_eff_probs
        if tox_cutoff is not None:
            schema_kwargs["tox_cutoff"] = tox_cutoff
        if eff_cutoff is not None:
            schema_kwargs["eff_cutoff"] = eff_cutoff
        if tox_certainty is not None:
            schema_kwargs["tox_certainty"] = tox_certainty
        if eff_certainty is not None:
            schema_kwargs["eff_certainty"] = eff_certainty
        if max_size is not None:
            schema_kwargs["max_size"] = max_size
        if first_dose is not None:
            schema_kwargs["first_dose"] = first_dose

        config = EffToxSchema(**schema_kwargs)
        first_dose = config.first_dose

        EfficacyToxicityDoseFindingTrial.__init__(
            self, first_dose, len(real_doses), config.max_size  # type: ignore
        )

        if theta_priors is None:
            if config.prior_tox_probs is None or config.prior_eff_probs is None:
                raise ValueError(
                    "Either theta_priors or both prior_tox_probs and prior_eff_probs must be provided."
                )
            theta_priors = efftox_priors_from_skeleton(
                real_doses, config.prior_tox_probs, config.prior_eff_probs
            )

        validate_efftox_priors(theta_priors, scale_doses(real_doses))

        self.real_doses = real_doses
        self._scaled_doses = scale_doses(real_doses)
        self.priors = theta_priors
        self.tox_cutoff = config.tox_cutoff
        self.eff_cutoff = config.eff_cutoff
        self.tox_certainty = config.tox_certainty
        self.eff_certainty = config.eff_certainty
        self.metric = metric
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation
        self.num_integral_steps = num_integral_steps
        self.epsilon = epsilon
        self.k_sd = k_sd
        self.max_iter = max_iter
        self.mass_threshold = mass_threshold

        self.reset()

    def _update_integrals(self, n: Any = None, rng: Any = None, **kwargs: Any) -> Any:
        """Recalculates integrals to update probabilities and utilities."""
        if rng is None:
            rng = self.rng
        if n is None:
            n = self.num_integral_steps
        cases = list(zip(self._doses, self._toxicities, self._efficacies))

        limit_args = {
            "k_sd": kwargs.get("k_sd", self.k_sd),
            "max_iter": kwargs.get("max_iter", self.max_iter),
            "mass_threshold": kwargs.get("mass_threshold", self.mass_threshold),
        }

        post_probs, _pds = efftox_get_posterior_probs(
            cases,
            self.priors,
            self._scaled_doses,
            self.tox_cutoff,
            self.eff_cutoff,
            n,
            self.epsilon,
            rng,
            **limit_args,
        )
        prob_tox, prob_eff, prob_acc_tox, prob_acc_eff = zip(*post_probs)
        admissable = np.array(
            [
                (x >= self.tox_certainty and y >= self.eff_certainty)
                or (i == self.maximum_dose_given() and x >= self.tox_certainty)
                for i, (x, y) in enumerate(zip(prob_acc_tox, prob_acc_eff))
            ]
        )
        admissable_set = [i + 1 for i, x in enumerate(admissable) if x]  # type: ignore[var-annotated]
        utility = np.array([self.metric(x[0], x[1]) for x in zip(prob_eff, prob_tox)])
        self.prob_tox = prob_tox
        self.prob_eff = prob_eff
        self.prob_acc_tox = prob_acc_tox
        self.prob_acc_eff = prob_acc_eff
        self._admissable_set = admissable_set
        self.utility = utility
        self.pds = _pds

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self, n: Any = None, rng: Any = None, **kwargs: Any) -> Any:
        if n is None:
            n = self.num_integral_steps
        self._update_integrals(n, rng, **kwargs)
        if self.treated_at_dose(self.first_dose()) > 0:
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            for i in np.argsort(-self.utility):  # type: ignore[operator]
                dose_level = i + 1
                if dose_level in self.admissable_set():
                    if (
                        self.avoid_skipping_untried_escalation
                        and max_dose_given
                        and dose_level - max_dose_given > 1
                    ):
                        pass
                    elif (
                        self.avoid_skipping_untried_deescalation
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
            if self.size() > 0:
                self._status = -10
            else:
                self._status = 0

        return self._next_dose

    def _EfficacyToxicityDoseFindingTrial__reset(self) -> Any:
        self.prob_tox = []  # type: ignore
        self.prob_eff = []  # type: ignore
        self.prob_acc_tox = []  # type: ignore
        self.prob_acc_eff = []  # type: ignore
        self._admissable_set = []
        self.utility = []  # type: ignore

    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def report(self) -> dict:  # type: ignore
        """Generates a standardized JSON-serializable report of the trial.

        Returns:
            collections.OrderedDict: The trial outcome report.
        """
        report = super().report()
        report.update(efftox_dtp_detail(self))
        return report

    def tabulate(self) -> Any:
        """Returns a pandas DataFrame summarising the trial.

        Returns:
            pandas.DataFrame: The trial summary.
        """
        df = EfficacyToxicityDoseFindingTrial.tabulate(self)
        df["P(Eff)"] = self.prob_eff
        df["P(Tox)"] = self.prob_tox
        df["P(AccEff)"] = self.prob_acc_eff
        df["P(AccTox)"] = self.prob_acc_tox
        df["Admissible"] = self.dose_admissability()
        df["Utility"] = self.utility
        return df

    def posterior_params(self, n: Any = None, rng: Any = None, **kwargs: Any) -> Any:
        """Gets the posterior parameter estimates.

        Args:
            n (int, optional): The number of points for Monte Carlo
                integration. Defaults to `None`.
            rng (numpy.random.Generator, optional): The random number generator.
            **kwargs: Additional arguments for limit refinement.

        Returns:
            list: A list of posterior parameter estimates.
        """
        if n is None:
            n = self.num_integral_steps
        cases = list(zip(self._doses, self._toxicities, self._efficacies))

        limit_args = {
            "k_sd": kwargs.get("k_sd", self.k_sd),
            "max_iter": kwargs.get("max_iter", self.max_iter),
            "mass_threshold": kwargs.get("mass_threshold", self.mass_threshold),
        }

        post_params, pds = efftox_get_posterior_params(
            cases, self.priors, self._scaled_doses, n, self.epsilon, rng, **limit_args
        )
        return post_params

    def optimal_decision(self, prob_tox: Any, prob_eff: Any) -> Any:
        """Determines the optimal biological dose.

        Args:
            prob_tox (list[float]): The probability of toxicity for each dose.
            prob_eff (list[float]): The probability of efficacy for each dose.

        Returns:
            int: The optimal biological dose.
        """
        admiss, u, _, obd, _ = solve_metrizable_efftox_scenario(
            prob_tox, prob_eff, self.metric, self.tox_cutoff, self.eff_cutoff
        )
        return obd

    def scaled_doses(self) -> Any:
        """Gets the scaled dose levels.

        Returns:
            numpy.ndarray: The scaled doses.
        """
        return self._scaled_doses


    def prob_superior_utility(self, dl1: Any, dl2: Any) -> Any:
        """Calculates the probability that one dose has superior utility over another.

        Args:
            dl1 (int): The first dose level (1-based).
            dl2 (int): The second dose level (1-based).

        Returns:
            float: The probability that dose `dl1` has superior utility to
                dose `dl2`.
        """
        if dl1 == dl2:
            return 0

        samp = self.pds._samp
        p = self.pds._probs
        p /= p.sum()

        x1 = self.scaled_doses()[dl1 - 1]
        x1_tox_probs = _pi_T(x1, mu=samp[:, 0], beta=samp[:, 1])
        x1_eff_probs = _pi_E(x1, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
        u1 = self.metric(x1_eff_probs, x1_tox_probs)

        x2 = self.scaled_doses()[dl2 - 1]
        x2_tox_probs = _pi_T(x2, mu=samp[:, 0], beta=samp[:, 1])
        x2_eff_probs = _pi_E(x2, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
        u2 = self.metric(x2_eff_probs, x2_tox_probs)

        return np.sum(p * (u1 > u2))

    def utility_superiority_matrix(self) -> Any:
        """Calculates the utility superiority matrix.

        Returns:
            numpy.ndarray: A matrix where element (i, j) is the
                probability that dose i has superior utility to dose j.
        """
        superiority_mat = np.zeros((self.num_doses, self.num_doses))
        superiority_mat[:] = np.nan  # type: ignore[index]
        for i in range(1, self.num_doses + 1):
            for j in range(i + 1, self.num_doses + 1):
                p = self.prob_superior_utility(i, j)
                superiority_mat[i - 1, j - 1] = p  # type: ignore[index]
                superiority_mat[j - 1, i - 1] = 1 - p  # type: ignore[index]
        return superiority_mat


def solve_metrizable_efftox_scenario(prob_tox: Any, prob_eff: Any, metric: Any, tox_cutoff: Any, eff_cutoff: Any) -> Any:
    """Solves a metrizable efficacy-toxicity dose-finding scenario.

    Args:
        prob_tox (list[float]): The probabilities of toxicity for each dose.
        prob_eff (list[float]): The probabilities of efficacy for each dose.
        metric (LpNormCurve | InverseQuadraticCurve): The utility metric.
        tox_cutoff (float): The maximum acceptable toxicity probability.
        eff_cutoff (float): The minimum acceptable efficacy probability.

    Returns:
        tuple: A tuple containing conformability, utilities, optimal
            utility, optimal dose, and utility cushion.
    """
    from clintrials.validation import validate_matching_lengths

    validate_matching_lengths(prob_tox=prob_tox, prob_eff=prob_eff)

    t = prob_tox
    r = prob_eff
    t = np.where(t <= 0, 0.001, t)
    t = np.where(t >= 1, 0.999, t)
    r = np.where(r <= 0, 0.001, r)
    r = np.where(r >= 1, 0.999, r)

    conform = np.array(
        [(eff >= eff_cutoff) and (tox <= tox_cutoff) for eff, tox in zip(r, t)]
    )
    util = np.array([metric(eff, tox) for eff, tox in zip(r, t)])
    conform_util = np.where(conform, util, -np.inf)

    if np.all(np.isnan(util)):
        logging.warning("All NaN util encountered in solve_metrizable_efftox_scenario")
        return conform, util, np.nan, -1, np.nan
    elif np.all(np.isnan(conform_util)):
        logging.warning(
            "All NaN conform_util encountered in solve_metrizable_efftox_scenario"
        )
        return conform, util, np.nan, -1, np.nan
    else:
        if sum(conform) >= 2:
            obd = np.nanargmax(conform_util) + 1
            u2, u1 = np.sort(conform_util)[-2:]
            u_cushion = u1 - u2
            return conform, util, u1, obd, u_cushion
        elif sum(conform) >= 1:
            obd = np.nanargmax(conform_util) + 1
            u1 = np.nanmax(conform_util)
            return conform, util, u1, obd, np.nan

    return conform, util, np.nan, -1, np.nan


def efftox_dtp_detail(trial: Any) -> Any:
    """Gets EffTox-specific details for DTP reporting.

    Args:
        trial (EffTox): An instance of the EffTox class.

    Returns:
        collections.OrderedDict: A dictionary with EffTox-specific details.
    """
    to_return = OrderedDict()

    to_return["Utility"] = iterable_to_json(trial.utility)  # type: ignore
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"Utility{dl}"] = trial.utility[i]

    to_return["ProbEff"] = iterable_to_json(trial.prob_eff)  # type: ignore
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbEff{dl}"] = trial.prob_eff[i]

    to_return["ProbAccEff"] = iterable_to_json(trial.prob_acc_eff)  # type: ignore
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbAccEff{dl}"] = trial.prob_acc_eff[i]

    to_return["ProbTox"] = iterable_to_json(trial.prob_tox)  # type: ignore
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbTox{dl}"] = trial.prob_tox[i]

    to_return["ProbAccTox"] = iterable_to_json(trial.prob_acc_tox)  # type: ignore
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbAccTox{dl}"] = trial.prob_acc_tox[i]

    sup_mat = trial.utility_superiority_matrix()
    to_return["SuperiorityMatrix"] = [iterable_to_json(x) for x in sup_mat]  # type: ignore

    obd = trial.next_dose()
    if obd > 0:
        min_sup = np.nanmin(sup_mat[obd - 1])
    else:
        min_sup = np.nan
    to_return["MinProbSuperiority"] = atomic_to_json(min_sup)  # type: ignore

    return to_return


__all__ = [
    "EffTox",
    "LpNormCurve",
    "InverseQuadraticCurve",
    "efftox_dtp_detail",
    "solve_metrizable_efftox_scenario",
    "efftox_priors_from_skeleton",
    "validate_efftox_priors",
    "efftox_get_posterior_params",
    "efftox_get_posterior_probs",
    "scale_doses",
]


# Inject module-level docstring
if __doc__:
    from clintrials.core.registry import CORE_REGISTRY
    __doc__ = __doc__.format(**CORE_REGISTRY)
