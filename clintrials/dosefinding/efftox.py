"""
An implementation of Thall & Cook's EffTox design for dose-finding in
clinical trials.
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


import logging
from collections import OrderedDict

import numpy as np
from scipy.optimize import brentq

from clintrials.core.math import inverse_logit
from clintrials.core.stats import ProbabilityDensitySample
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial
from clintrials.utils import atomic_to_json, iterable_to_json


def scale_doses(real_doses):
    """Scales the doses for use in the EffTox model.

    Args:
        real_doses (list[float]): The actual dose amounts.

    Returns:
        numpy.ndarray: The scaled doses.
    """
    return np.log(real_doses) - np.mean(np.log(real_doses))


def _eta_T(scaled_dose, mu, beta):
    """Calculates the linear predictor for toxicity.

    Args:
        scaled_dose (float): The scaled dose.
        mu (float): The intercept parameter.
        beta (float): The slope parameter.

    Returns:
        float: The linear predictor for toxicity.
    """
    return mu + beta * scaled_dose


def _eta_E(scaled_dose, mu, beta1, beta2):
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


def _pi_T(scaled_dose, mu, beta):
    """Calculates the probability of toxicity.

    Args:
        scaled_dose (float): The scaled dose.
        mu (float): The intercept parameter.
        beta (float): The slope parameter.

    Returns:
        float: The probability of toxicity.
    """
    return inverse_logit(_eta_T(scaled_dose, mu, beta))


def _pi_E(scaled_dose, mu, beta1, beta2):
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


def _pi_ab(scaled_dose, tox, eff, mu_T, beta_T, mu_E, beta1_E, beta2_E, psi):
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
    a, b = eff, tox
    p1 = _pi_E(scaled_dose, mu_E, beta1_E, beta2_E)
    p2 = _pi_T(scaled_dose, mu_T, beta_T)
    response = p1**a * (1 - p1) ** (1 - a) * p2**b * (1 - p2) ** (1 - b)
    response += (
        -(1 ** (a + b))
        * p1
        * (1 - p1)
        * p2
        * (1 - p2)
        * (np.exp(psi) - 1)
        / (np.exp(psi) + 1)
    )
    return response


def _L_n(D, mu_T, beta_T, mu_E, beta1_E, beta2_E, psi):
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


def efftox_get_posterior_probs(
    cases, priors, scaled_doses, tox_cutoff, eff_cutoff, n=10**5, epsilon=1e-6
):
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

    Returns:
        tuple[list, ProbabilityDensitySample]: A tuple containing a list of
            posterior probabilities for each dose, and the
            `ProbabilityDensitySample` object.
    """
    if len(priors) != 6:
        raise ValueError("priors should have 6 items.")

    # Convert dose-levels given to dose amounts given
    if len(cases) > 0:
        dose_levels, tox_events, eff_events = zip(*cases)
        scaled_doses_given = [scaled_doses[x - 1] for x in dose_levels]
        _cases = list(zip(scaled_doses_given, tox_events, eff_events))
    else:
        _cases = []

    limits = [(dist.ppf(epsilon), dist.ppf(1 - epsilon)) for dist in priors]
    samp = np.column_stack(
        [np.random.uniform(*limit_pair, size=n) for limit_pair in limits]
    )

    lik_integrand = (
        lambda x: _L_n(_cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])
        * priors[0].pdf(x[:, 0])
        * priors[1].pdf(x[:, 1])
        * priors[2].pdf(x[:, 2])
        * priors[3].pdf(x[:, 3])
        * priors[4].pdf(x[:, 4])
        * priors[5].pdf(x[:, 5])
    )
    pds = ProbabilityDensitySample(samp, lik_integrand)

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


def efftox_get_posterior_params(cases, priors, scaled_doses, n=10**5, epsilon=1e-6):
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

    Returns:
        tuple[list, ProbabilityDensitySample]: A tuple containing a list of
            posterior parameter estimates and the `ProbabilityDensitySample`
            object.
    """
    if len(priors) != 6:
        raise ValueError("priors should have 6 items.")

    # Convert dose-levels given to dose amounts given
    if len(cases) > 0:
        dose_levels, tox_events, eff_events = zip(*cases)
        scaled_doses_given = [scaled_doses[x - 1] for x in dose_levels]
        _cases = zip(scaled_doses_given, tox_events, eff_events)
    else:
        _cases = []

    limits = [(dist.ppf(epsilon), dist.ppf(1 - epsilon)) for dist in priors]
    samp = np.column_stack(
        [np.random.uniform(*limit_pair, size=n) for limit_pair in limits]
    )

    lik_integrand = (
        lambda x: _L_n(_cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])
        * priors[0].pdf(x[:, 0])
        * priors[1].pdf(x[:, 1])
        * priors[2].pdf(x[:, 2])
        * priors[3].pdf(x[:, 3])
        * priors[4].pdf(x[:, 4])
        * priors[5].pdf(x[:, 5])
    )
    pds = ProbabilityDensitySample(samp, lik_integrand)

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

    def __init__(
        self,
        minimum_tolerable_efficacy,
        maximum_tolerable_toxicity,
        hinge_prob_eff,
        hinge_prob_tox,
    ):
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

        def _find_p(p):
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

    def __call__(self, prob_eff, prob_tox):
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

    def solve(self, prob_eff=None, prob_tox=None, delta=0):
        """Solves for one probability given the other and a utility delta.

        Args:
            prob_eff (float, optional): The probability of efficacy.
            prob_tox (float, optional): The probability of toxicity.
            delta (float, optional): The utility delta. Defaults to 0.

        Returns:
            float: The solved probability.
        """
        if prob_eff is None and prob_tox is None:
            raise Exception("Specify prob_eff or prob_tox")
        if prob_eff is not None and prob_tox is not None:
            raise Exception("Specify just one of prob_eff and prob_tox")

        x, y = prob_eff, prob_tox
        x_l, y_l = self.minimum_tolerable_efficacy, self.maximum_tolerable_toxicity
        scaled_delta = (1 - delta) ** self.p
        if x is None:
            # Solve for x
            b = y / y_l
            a = (scaled_delta - b**self.p) ** (1 / self.p)
            return 1 - (1 - x_l) * a
        else:
            # Solve for y
            a = (1 - x) / (1 - x_l)
            b_term = scaled_delta - a**self.p
            if b_term <= 0:
                return complex(np.nan, np.nan)
            b = b_term ** (1 / self.p)
            return b * y_l

    def get_tox(self, eff, util=0.0):
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

    def plot_contours(
        self,
        use_ggplot=False,
        prob_eff=None,
        prob_tox=None,
        n=1000,
        util_lower=-0.8,
        util_upper=0.8,
        util_delta=0.2,
        title="EffTox utility contours",
        custom_points_label="priors",
    ):
        """Plots the utility contours.

        Args:
            use_ggplot (bool, optional): If `True`, uses ggplot. Defaults to
                `False`.
            prob_eff (list[float], optional): A list of efficacy
                probabilities to plot as points. Defaults to `None`.
            prob_tox (list[float], optional): A list of toxicity
                probabilities to plot as points. Defaults to `None`.
            n (int, optional): The number of points per line. Defaults to 1000.
            util_lower (float, optional): The lowest utility value to plot.
                Defaults to -0.8.
            util_upper (float, optional): The highest utility value to plot.
                Defaults to 0.8.
            util_delta (float, optional): The increment for utility contours.
                Defaults to 0.2.
            title (str, optional): The chart title. Defaults to "EffTox
                utility contours".
            custom_points_label (str, optional): The label for the custom
                points. Defaults to "priors".

        Returns:
            A plot object.
        """
        eff_vals = np.linspace(0, 1, n)
        util_vals = np.linspace(
            util_lower, util_upper, ((util_upper - util_lower) / util_delta) + 1
        )

        if use_ggplot:
            raise NotImplementedError()
        else:
            import matplotlib.pyplot as plt

            # Plot general contours
            for u in util_vals:
                tox_vals = [self.get_tox(eff=x, util=u) for x in eff_vals]
                plt.plot(eff_vals, tox_vals, "-", c="k", lw=0.5)

            # Add neutral utility contour
            tox_vals = [self.get_tox(eff=x, util=0) for x in eff_vals]
            plt.plot(eff_vals, tox_vals, "-", c="k", lw=2, label="neutral utility")

            # Add hinge points
            hinge_prob_eff, hinge_prob_tox = zip(*self.hinge_points)
            plt.plot(hinge_prob_eff, hinge_prob_tox, "ro", ms=10, label="hinge points")

            # Add custom points
            if prob_eff is not None and prob_tox is not None:
                plt.plot(prob_eff, prob_tox, "b^", ms=10, label=custom_points_label)

            # Plot size
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel("Prob(Efficacy)")
            plt.ylabel("Prob(Toxicity)")
            plt.title(title)
            plt.legend()

            # Return
            p = plt.gcf()
            phi = (np.sqrt(5) + 1) / 2.0
            p.set_size_inches(12, 12 / phi)


class InverseQuadraticCurve:
    """Fits an indifference contour using an inverse quadratic curve."""

    def __init__(self, points):
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
        if sum(np.abs(f(x) - y)) > 0.00001:
            raise ValueError("%s do not fit an ABC curve well" % points)
        self.f = f
        self.a, self.b, self.c = a, b, c

    def __call__(self, prob_eff, prob_tox):
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

            def intersection_expression(x, m, f):
                return m * (x - 1) - f(x)

            x_00 = brentq(intersection_expression, 0.0001, 1, args=(gradient, self.f))
            y_00 = self.f(x_00)
            d1 = np.sqrt((x_00 - 1) ** 2 + y_00**2)
            d2 = np.sqrt((x - 1) ** 2 + y**2)

            return d1 / d2 - 1
        else:
            return np.nan

    def solve(self, prob_eff=None, prob_tox=None, delta=0):
        """Solves for one probability given the other and a utility delta.

        Args:
            prob_eff (float, optional): The probability of efficacy.
            prob_tox (float, optional): The probability of toxicity.
            delta (float, optional): The utility delta. Defaults to 0.

        Returns:
            float: The solved probability.
        """
        if delta != 0:
            raise NotImplementedError("Only contours for delta=0 are supported.")

        if prob_eff is None and prob_tox is None:
            raise Exception("Specify prob_eff or prob_tox")
        if prob_eff is not None and prob_tox is not None:
            raise Exception("Specify just one of prob_eff and prob_tox")

        if prob_eff is not None:
            return self.f(prob_eff)
        else:
            A = prob_tox - self.a
            B = -self.b
            C = -self.c

            if A == 0:
                if B == 0:
                    return np.nan
                else:
                    x = -C / B
                    return x if 0 < x < 1 else np.nan

            discriminant = B**2 - 4 * A * C
            if discriminant < 0:
                return np.nan

            x1 = (-B + np.sqrt(discriminant)) / (2 * A)
            x2 = (-B - np.sqrt(discriminant)) / (2 * A)

            if 0 < x1 < 1:
                return x1
            elif 0 < x2 < 1:
                return x2
            else:
                return np.nan

    def plot_contours(
        self,
        use_ggplot=False,
        prior_eff_probs=None,
        prior_tox_probs=None,
        n=1000,
        util_lower=-0.8,
        util_upper=0.8,
        util_delta=0.2,
        title="EffTox utility contours",
    ):
        """Plots the utility contours."""
        raise NotImplementedError()


HingedCurve = LpNormCurve
ABC_Curve = InverseQuadraticCurve


class EffTox(EfficacyToxicityDoseFindingTrial):
    """An object-oriented implementation of Thall & Cook's EffTox trial design.

    See Thall, P.F. & Cook, J.D. (2004) - Dose-Finding Based on
    Efficacy-Toxicity Trade-Offs.
    """

    def __init__(
        self,
        real_doses,
        theta_priors,
        tox_cutoff,
        eff_cutoff,
        tox_certainty,
        eff_certainty,
        metric,
        max_size,
        first_dose=1,
        avoid_skipping_untried_escalation=True,
        avoid_skipping_untried_deescalation=True,
        num_integral_steps=10**5,
        epsilon=1e-6,
    ):
        """Initializes an EffTox trial object.

        Args:
            real_doses (list[float]): A list of the actual dose amounts.
            theta_priors (list): A list of 6 prior distributions for the
                model parameters.
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
            first_dose (int, optional): The starting dose level (1-based).
                Defaults to 1.
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

        Raises:
            ValueError: If `theta_priors` does not have 6 items.
        """
        EfficacyToxicityDoseFindingTrial.__init__(
            self, first_dose, len(real_doses), max_size
        )

        if len(theta_priors) != 6:
            raise ValueError("theta_priors should have 6 items.")

        self.real_doses = real_doses
        self._scaled_doses = np.log(real_doses) - np.mean(np.log(real_doses))
        self.priors = theta_priors
        self.tox_cutoff = tox_cutoff
        self.eff_cutoff = eff_cutoff
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty
        self.metric = metric
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation
        self.num_integral_steps = num_integral_steps
        self.epsilon = epsilon

        self.reset()

    def _update_integrals(self, n=None):
        """Recalculates integrals to update probabilities and utilities."""
        if n is None:
            n = self.num_integral_steps
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        post_probs, _pds = efftox_get_posterior_probs(
            cases,
            self.priors,
            self._scaled_doses,
            self.tox_cutoff,
            self.eff_cutoff,
            n,
            self.epsilon,
        )
        prob_tox, prob_eff, prob_acc_tox, prob_acc_eff = zip(*post_probs)
        admissable = np.array(
            [
                (
                    x >= self.tox_certainty and y >= self.eff_certainty
                )
                or (
                    i == self.maximum_dose_given() and x >= self.tox_certainty
                )
                for i, (x, y) in enumerate(zip(prob_acc_tox, prob_acc_eff))
            ]
        )
        admissable_set = [i + 1 for i, x in enumerate(admissable) if x]
        utility = np.array([self.metric(x[0], x[1]) for x in zip(prob_eff, prob_tox)])
        self.prob_tox = prob_tox
        self.prob_eff = prob_eff
        self.prob_acc_tox = prob_acc_tox
        self.prob_acc_eff = prob_acc_eff
        self._admissable_set = admissable_set
        self.utility = utility
        self.pds = _pds

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self, n=None):
        if n is None:
            n = self.num_integral_steps
        self._update_integrals(n)
        if self.treated_at_dose(self.first_dose()) > 0:
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            for i in np.argsort(
                -self.utility
            ):
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

    def _EfficacyToxicityDoseFindingTrial__reset(self):
        self.prob_tox = []
        self.prob_eff = []
        self.prob_acc_tox = []
        self.prob_acc_eff = []
        self._admissable_set = []
        self.utility = []

    def has_more(self):
        """Checks if the trial is ongoing.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def tabulate(self):
        df = EfficacyToxicityDoseFindingTrial.tabulate(self)
        df["P(Eff)"] = self.prob_eff
        df["P(Tox)"] = self.prob_tox
        df["P(AccEff)"] = self.prob_acc_eff
        df["P(AccTox)"] = self.prob_acc_tox
        df["Admissible"] = self.dose_admissability()
        df["Utility"] = self.utility
        return df

    def posterior_params(self, n=None):
        """Gets the posterior parameter estimates.

        Args:
            n (int, optional): The number of points for Monte Carlo
                integration. Defaults to `None`.

        Returns:
            list: A list of posterior parameter estimates.
        """
        if n is None:
            n = self.num_integral_steps
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        post_params, pds = efftox_get_posterior_params(
            cases, self.priors, self._scaled_doses, n, self.epsilon
        )
        return post_params

    def optimal_decision(self, prob_tox, prob_eff):
        admiss, u, u_star, obd, u_cushion = solve_metrizable_efftox_scenario(
            prob_tox, prob_eff, self.metric, self.tox_cutoff, self.eff_cutoff
        )
        return obd

    def scaled_doses(self):
        """Gets the scaled dose levels.

        Returns:
            numpy.ndarray: The scaled doses.
        """
        return self._scaled_doses

    def _post_density_plot(
        self, func=None, x_name="", plot_title="", include_doses=None, boot_samps=1000
    ):
        import pandas as pd
        from ggplot import aes, geom_density, ggplot, ggtitle

        if include_doses is None:
            include_doses = range(1, self.num_doses + 1)

        def my_func(x, samp):
            tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
            eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
            u = self.metric(eff_probs, tox_probs)
            return u

        if func is None:
            func = my_func

        x_boot = []
        dose_indices = []
        samp = self.pds._samp
        p = self.pds._probs
        p /= p.sum()
        for i, x in enumerate(self.scaled_doses()):
            dose_index = i + 1
            if dose_index in include_doses:
                x = func(x, samp)
                x_boot.extend(np.random.choice(x, size=boot_samps, replace=True, p=p))
                dose_indices.extend(np.repeat(dose_index, boot_samps))
        df = pd.DataFrame({x_name: x_boot, "Dose": dose_indices})
        return (
            ggplot(aes(x=x_name, fill="Dose"), data=df)
            + geom_density(alpha=0.6)
            + ggtitle(plot_title)
        )

    def plot_posterior_tox_prob_density(self, include_doses=None, boot_samps=1000):
        """Plots the posterior densities of the toxicity probabilities.

        Args:
            include_doses (list[int], optional): A list of dose levels to
                include. Defaults to all doses.
            boot_samps (int, optional): The number of bootstrap samples.
                Defaults to 1000.

        Returns:
            A ggplot object.
        """
        def get_prob_tox(x, samp):
            tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
            return tox_probs

        return self._post_density_plot(
            func=get_prob_tox,
            x_name="Prob(Toxicity)",
            plot_title="Posterior densities of Prob(Toxicity)",
            include_doses=include_doses,
            boot_samps=boot_samps,
        )

    def plot_posterior_eff_prob_density(self, include_doses=None, boot_samps=1000):
        """Plots the posterior densities of the efficacy probabilities.

        Args:
            include_doses (list[int], optional): A list of dose levels to
                include. Defaults to all doses.
            boot_samps (int, optional): The number of bootstrap samples.
                Defaults to 1000.

        Returns:
            A ggplot object.
        """
        def get_prob_eff(x, samp):
            eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
            return eff_probs

        return self._post_density_plot(
            func=get_prob_eff,
            x_name="Prob(Efficacy)",
            plot_title="Posterior densities of Prob(Efficacy)",
            include_doses=include_doses,
            boot_samps=boot_samps,
        )

    def plot_posterior_utility_density(self, include_doses=None, boot_samps=1000):
        """Plots the posterior densities of the dose utilities.

        Args:
            include_doses (list[int], optional): A list of dose levels to
                include. Defaults to all doses.
            boot_samps (int, optional): The number of bootstrap samples.
                Defaults to 1000.

        Returns:
            A ggplot object.
        """
        def get_utility(x, samp):
            tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
            eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
            u = self.metric(eff_probs, tox_probs)
            return u

        return self._post_density_plot(
            func=get_utility,
            x_name="Utility",
            plot_title="Posterior densities of Utility",
            include_doses=include_doses,
            boot_samps=boot_samps,
        )

    def prob_superior_utility(self, dl1, dl2):
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

    def utility_superiority_matrix(self):
        """Calculates the utility superiority matrix.

        Returns:
            numpy.ndarray: A matrix where element (i, j) is the
                probability that dose i has superior utility to dose j.
        """
        superiority_mat = np.zeros((4, 4))
        superiority_mat[:] = np.nan
        for i in range(1, self.num_doses + 1):
            for j in range(i + 1, self.num_doses + 1):
                p = self.prob_superior_utility(i, j)
                superiority_mat[i - 1, j - 1] = p
                superiority_mat[j - 1, i - 1] = 1 - p
        return superiority_mat


def solve_metrizable_efftox_scenario(
    prob_tox, prob_eff, metric, tox_cutoff, eff_cutoff
):
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
    if len(prob_tox) != len(prob_eff):
        raise Exception(
            "prob_tox and prob_eff should be lists or tuples of the same length."
        )

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


def get_obd(tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    """Gets the optimal biologically-effective dose (OBD)."""
    X = solve_metrizable_efftox_scenario(
        tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff
    )
    conform, util, u_star, obd, u_cushion = X
    return obd


def get_conformative_doses(tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    """Gets the set of conformative doses."""
    X = solve_metrizable_efftox_scenario(
        tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff
    )
    conform, util, u_star, obd, u_cushion = X
    return [int(x) for x in conform]


def get_util(tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    """Gets the utilities of the doses."""
    X = solve_metrizable_efftox_scenario(
        tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff
    )
    conform, util, u_star, obd, u_cushion = X
    return np.round(util, 2)


def classify_problem(
    delta, prob_tox, prob_eff, metric, tox_cutoff, eff_cutoff, text_label=True
):
    """Classifies the dose-finding problem."""
    X = solve_metrizable_efftox_scenario(
        prob_tox, prob_eff, metric, tox_cutoff, eff_cutoff
    )
    conform, util, u_star, obd, u_cushion = X
    dose_within_delta = np.array([u >= (1 - delta) * u_star for u in util])
    if obd == -1:
        return "Stop" if text_label else 1
    elif sum(dose_within_delta) == 1:
        return "Optimal" if text_label else 2
    else:
        return "Desirable" if text_label else 3


def get_problem_class(delta, tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    """Gets the class of the dose-finding problem."""
    return classify_problem(delta, tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff)


def classify_tox_class(prob_tox, tox_cutoff, text_label=True):
    """Classifies the toxicity profile."""
    prob_tox = np.array(prob_tox)
    if sum(prob_tox < tox_cutoff) == len(prob_tox):
        return "Tolerable" if text_label else 1
    elif sum(prob_tox > tox_cutoff) == len(prob_tox):
        return "Toxic" if text_label else 2
    else:
        return "Mixed" if text_label else 3


def get_tox_class(tox_curve, tox_cutoff):
    """Gets the toxicity class."""
    return classify_tox_class(tox_curve, tox_cutoff)


def classify_eff_class(prob_eff, eff_cutoff, text_label=True):
    """Classifies the efficacy profile."""
    prob_eff = np.array(prob_eff)
    max_eff = np.max(prob_eff)
    if np.all([prob_eff[i] > prob_eff[i - 1] for i in range(1, len(prob_eff))]):
        return "Monotonic" if text_label else 1
    elif sum(prob_eff == max_eff) == 1:
        return "Unimodal" if text_label else 2
    elif sum(prob_eff == max_eff) > 1:
        return "Plateau" if text_label else 3
    else:
        return "Weird" if text_label else 4


def get_eff_class(eff_curve, eff_cutoff):
    """Gets the efficacy class."""
    return classify_eff_class(eff_curve, eff_cutoff)


def efftox_dtp_detail(trial):
    """Gets EffTox-specific details for DTP reporting.

    Args:
        trial (EffTox): An instance of the EffTox class.

    Returns:
        collections.OrderedDict: A dictionary with EffTox-specific details.
    """
    to_return = OrderedDict()

    to_return["Utility"] = iterable_to_json(trial.utility)
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"Utility{dl}"] = trial.utility[i]

    to_return["ProbEff"] = iterable_to_json(trial.prob_eff)
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbEff{dl}"] = trial.prob_eff[i]

    to_return["ProbAccEff"] = iterable_to_json(trial.prob_acc_eff)
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbAccEff{dl}"] = trial.prob_acc_eff[i]

    to_return["ProbTox"] = iterable_to_json(trial.prob_tox)
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbTox{dl}"] = trial.prob_tox[i]

    to_return["ProbAccTox"] = iterable_to_json(trial.prob_acc_tox)
    for i, dl in enumerate(trial.dose_levels()):
        to_return[f"ProbAccTox{dl}"] = trial.prob_acc_tox[i]

    sup_mat = trial.utility_superiority_matrix()
    to_return["SuperiorityMatrix"] = [iterable_to_json(x) for x in sup_mat]

    obd = trial.next_dose()
    if obd > 0:
        min_sup = np.nanmin(sup_mat[obd - 1])
    else:
        min_sup = np.nan
    to_return["MinProbSuperiority"] = atomic_to_json(min_sup)

    return to_return


__all__ = [
    "EffTox",
    "LpNormCurve",
    "InverseQuadraticCurve",
    "efftox_dtp_detail",
    "solve_metrizable_efftox_scenario",
]
