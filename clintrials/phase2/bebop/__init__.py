__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

__all__ = ["peps2v1", "peps2v2"]

"""

 BeBOP: Bayesian design with Bivariate Outcomes and Predictive variables
 Brock, et al. To be published.

 BeBOP studies the dual primary outcomes efficacy and toxicity.
 The two events can be associated to reflect the potential for correlated
 outcomes. The design models the probabilities of efficacy and toxicity
 using logistic models so that the information in predictive variables
 can be incorporated to tailor the treatment acceptance / rejection
 decision.

 This is a generalisation of the design that was used in the PePS2 trial.
 PePS2 studies the efficacy and toxicity of a drug in a population of
 performance status 2 lung cancer patients. Patient outcomes may plausibly
 be effected by whether or not they have been treated before, and the
 expression rate of PD-L1 in their cells.

 PePS2 uses Brock et al's BeBOP design to incorporate the potentially
 predictive data in PD-L1 expression rate and whether or not a patient has
 been pre-treated to find the sub-population(s) where the drug works
 and is tolerable.

"""

import numpy
import pandas as pd

from clintrials.core.stats import ProbabilityDensitySample, correlation_ci


class BeBOP:
    """BeBOP: Bayesian design with Bivariate Outcomes and Predictive variables."""

    def __init__(self, theta_priors, efficacy_model, toxicity_model, joint_model):
        """Initializes a BeBOP model.

        Args:
            theta_priors (list): List of prior distributions for elements of parameter vector, theta.
                Each prior object should support obj.ppf(x) and obj.pdf(x).
            efficacy_model (callable): Function with signature x, theta. Returns probability of efficacy.
            toxicity_model (callable): Function with signature x, theta. Returns probability of toxicity.
            joint_model (callable): Function with signature x, theta. Returns joint probability.
        """

        self.priors = theta_priors
        self._pi_e = efficacy_model
        self._pi_t = toxicity_model
        self._pi_ab = joint_model
        # Initialise model
        self.reset()

    def reset(self):
        """Resets the model state."""
        self.cases = []
        self._pds = None

    def _l_n(self, D, theta):
        if len(D) > 0:
            lik = numpy.array([self._pi_ab(x, theta) for x in D])
            return lik.prod(axis=0)
        else:
            return numpy.ones(len(theta))

    def size(self):
        """Returns the number of observed cases.

        Returns:
            int: Number of cases.
        """
        return len(self.cases)

    def efficacies(self):
        """Returns a list of efficacy outcomes for all observed cases.

        Returns:
            list: Efficacy outcomes.
        """
        return [case[0] for case in self.cases]

    def toxicities(self):
        """Returns a list of toxicity outcomes for all observed cases.

        Returns:
            list: Toxicity outcomes.
        """
        return [case[1] for case in self.cases]

    def get_case_elements(self, i):
        """Returns a list of the i-th element of all observed cases.

        Args:
            i (int): The index of the element.

        Returns:
            list: The i-th elements.
        """
        return [case[i] for case in self.cases]

    def update(self, cases, n=10**6, epsilon=0.00001, **kwargs):
        """
        Update the model with new observed cases.

        This method updates the posterior distribution of the model parameters
        based on new data. It uses Monte Carlo integration to approximate the
        posterior. The posterior is stored as a `ProbabilityDensitySample`
        object in `self._pds`.

        Parameters
        ----------
        cases : list of list
            A list of case vectors. Each vector represents a patient and should
            contain outcome variables followed by predictor variables.
            For example, in PePS2 version 2, each case is [eff, tox, pre-treated,
            low_pdl1, mid_pdl1].
        n : int, optional
            The number of samples to use for the Monte Carlo integration.
            Should be a positive integer. Defaults to 1,000,000.
        epsilon : float, optional
            A small value in (0, 1) used to determine integration limits via
            the percent point function (ppf) of the priors.
            Defaults to 0.00001.
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `n` is not a positive integer or `epsilon` is not between 0 and 1.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import norm
        >>> from clintrials.phase2.bebop import BeBOP
        >>> from clintrials.phase2.bebop.peps2v2 import pi_e, pi_t, pi_ab
        >>> priors = [norm(0, 2)] * 6
        >>> model = BeBOP(priors, pi_e, pi_t, pi_ab)
        >>> cases = [[1, 0, 0, 0, 0], [0, 1, 1, 1, 0]]
        >>> model.update(cases, n=1000)
        >>> model.size()
        2
        """
        from clintrials.validation import (
            validate_bounds,
            validate_positive_integer,
        )

        validate_positive_integer(n, "n")
        validate_bounds(epsilon, 0, 1, "epsilon", exclusive=True)

        self.cases.extend(cases)
        limits = [(dist.ppf(epsilon), dist.ppf(1 - epsilon)) for dist in self.priors]
        lik_integrand = lambda x: self._l_n(cases, x) * numpy.prod(
            numpy.array([dist.pdf(col) for (dist, col) in zip(self.priors, x.T)]),
            axis=0,
        )
        from clintrials.core.numerics import adaptive_mc_integration

        refined_limits, pds = adaptive_mc_integration(
            lik_integrand,
            limits,
            n=n,
            max_iter=1,  # Keep non-iterative logic similar to original, but centralized
        )
        self._pds = pds
        return

    def _predict_case(self, case, eff_cutoff, tox_cutoff, pds, samp, estimate_ci=False):
        x = case
        eff_probs = self._pi_e(x, samp)
        tox_probs = self._pi_t(x, samp)
        from collections import OrderedDict

        predictions = OrderedDict(
            [
                ("Pr(Eff)", pds.expectation(eff_probs)),
                ("Pr(Tox)", pds.expectation(tox_probs)),
                ("Pr(AccEff)", pds.expectation(eff_probs > eff_cutoff)),
                ("Pr(AccTox)", pds.expectation(tox_probs < tox_cutoff)),
            ]
        )

        if estimate_ci:
            predictions["Pr(Eff) Lower"] = pds.quantile_vector(
                eff_probs, 0.05, start_value=0.05
            )
            predictions["Pr(Eff) Upper"] = pds.quantile_vector(
                eff_probs, 0.95, start_value=0.95
            )
            predictions["Pr(Tox) Lower"] = pds.quantile_vector(
                tox_probs, 0.05, start_value=0.05
            )
            predictions["Pr(Tox) Upper"] = pds.quantile_vector(
                tox_probs, 0.95, start_value=0.95
            )
        return predictions

    def predict(
        self, cases, eff_cutoff, tox_cutoff, to_pandas=False, estimate_ci=False
    ):
        """Predicts the probability of efficacy and toxicity for given cases.

        Args:
            cases (list): A list of cases to predict.
            eff_cutoff (float): The minimum acceptable efficacy probability.
            tox_cutoff (float): The maximum acceptable toxicity probability.
            to_pandas (bool, optional): If True, return results as a pandas DataFrame. Defaults to False.
            estimate_ci (bool, optional): If True, calculate lower and upper confidence intervals. Defaults to False.

        Returns:
            list or pandas.DataFrame: The predictions.
        """
        if self._pds is not None:
            pds = self._pds
            samp = pds._samp
            fitted = [
                self._predict_case(
                    x, eff_cutoff, tox_cutoff, pds, samp, estimate_ci=estimate_ci
                )
                for x in cases
            ]
            if to_pandas:
                if estimate_ci:
                    return pd.DataFrame(
                        fitted,
                        columns=[
                            "Pr(Eff)",
                            "Pr(Tox)",
                            "Pr(AccEff)",
                            "Pr(AccTox)",
                            "Pr(Eff) Lower",
                            "Pr(Eff) Upper",
                            "Pr(Tox) Lower",
                            "Pr(Tox) Upper",
                        ],
                    )
                else:
                    return pd.DataFrame(
                        fitted,
                        columns=["Pr(Eff)", "Pr(Tox)", "Pr(AccEff)", "Pr(AccTox)"],
                    )
            else:
                return fitted
        else:
            return None

    def get_posterior_param_means(self):
        """Returns the posterior means of the model parameters.

        Returns:
            list: Posterior means of parameters.
        """
        if self._pds:
            return numpy.apply_along_axis(
                lambda x: self._pds.expectation(x), 0, self._pds._samp
            )
        else:
            return []

    def theta_estimate(self, i, alpha=0.05):
        """Get posterior confidence interval and mean estimate of element i in parameter vector.

        Args:
            i (int): Index of the parameter.
            alpha (float, optional): The significance level. Defaults to 0.05.

        Returns:
            numpy.ndarray: Array containing lower bound, mean estimate, and upper bound.
        """

        if i < len(self.priors):
            mu = self._pds.expectation(self._pds._samp[:, i])
            return numpy.array(
                [
                    self._pds.quantile(i, alpha / 2),
                    mu,
                    self._pds.quantile(i, 1 - alpha / 2),
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
        psi_samples = self._pds._samp[:, 5]
        correlation_samples = (numpy.exp(psi_samples) - 1) / (
            numpy.exp(psi_samples) + 1
        )

        return correlation_ci(
            samples=correlation_samples,
            weights=self._pds._probs,
            alpha=alpha,
            method="bayes",
        )
