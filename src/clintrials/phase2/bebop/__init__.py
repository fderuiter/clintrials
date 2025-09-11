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

from clintrials.core.stats import ProbabilityDensitySample


class BeBOP:
    """A class for the BeBOP trial design."""

    def __init__(self, theta_priors, efficacy_model, toxicity_model, joint_model):
        """Initializes the BeBOP trial.

        Args:
            theta_priors: A list of prior distributions for the model
                parameters.
            efficacy_model: A function to calculate the probability of efficacy.
            toxicity_model: A function to calculate the probability of toxicity.
            joint_model: A function to calculate the joint probability of
                efficacy and toxicity.
        """

        self.priors = theta_priors
        self._pi_e = efficacy_model
        self._pi_t = toxicity_model
        self._pi_ab = joint_model
        # Initialise model
        self.reset()

    def reset(self):
        """Resets the trial to its initial state."""
        self.cases = []
        self._pds = None

    def _l_n(self, D, theta):
        if len(D) > 0:
            lik = numpy.array(map(lambda x: self._pi_ab(x, theta), D))
            return lik.prod(axis=0)
        else:
            return numpy.ones(len(theta))

    def size(self):
        """Gets the number of patients treated so far.

        Returns:
            The number of patients treated.
        """
        return len(self.cases)

    def efficacies(self):
        """Gets the efficacy outcome for each patient.

        Returns:
            A list of efficacy outcomes.
        """
        return [case[0] for case in self.cases]

    def toxicities(self):
        """Gets the toxicity outcome for each patient.

        Returns:
            A list of toxicity outcomes.
        """
        return [case[1] for case in self.cases]

    def get_case_elements(self, i):
        """Gets the i-th element for each case.

        Args:
            i: The index of the element to get.

        Returns:
            A list of the i-th elements.
        """
        return [case[i] for case in self.cases]

    def update(self, cases, n=10**6, epsilon=0.00001, **kwargs):
        """Update the model with new observed cases.

        This method updates the posterior distribution of the model parameters
        based on new data. It uses Monte Carlo integration to approximate the
        posterior. The posterior is stored as a `ProbabilityDensitySample`
        object in `self._pds`.

        :param cases: A list of case vectors. Each vector represents a patient
                      and should contain the outcome variables followed by the
                      predictor variables, as expected by the efficacy, toxicity,
                      and joint probability models.
        :type cases: list[list]
        :param n: The number of samples to use for the Monte Carlo integration.
                  A larger number will produce more accurate estimates but will
                  be slower.
        :type n: int
        :param epsilon: A small value used to determine the integration limits
                        by taking the `epsilon` and `1-epsilon` quantiles of
                        the prior distributions. This defines a hyperrectangle
                        from which to sample.
        :type epsilon: float
        :param kwargs: Not used.
        """
        self.cases.extend(cases)
        limits = [(dist.ppf(epsilon), dist.ppf(1 - epsilon)) for dist in self.priors]
        samp = numpy.column_stack(
            [numpy.random.uniform(*limit_pair, size=n) for limit_pair in limits]
        )
        lik_integrand = lambda x: self._l_n(cases, x) * numpy.prod(
            numpy.array([dist.pdf(col) for (dist, col) in zip(self.priors, x.T)]),
            axis=0,
        )
        self._pds = ProbabilityDensitySample(samp, lik_integrand)
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
        """Makes predictions for a given set of cases.

        Args:
            cases: A list of case vectors.
            eff_cutoff: The efficacy cutoff.
            tox_cutoff: The toxicity cutoff.
            to_pandas: If True, returns a pandas DataFrame.
            estimate_ci: If True, includes confidence intervals in the
                output.

        Returns:
            A list of predictions or a pandas DataFrame.
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
        """Gets the posterior means of the model parameters.

        Returns:
            A numpy array of the posterior means.
        """
        if self._pds:
            return numpy.apply_along_axis(
                lambda x: self._pds.expectation(x), 0, self._pds._samp
            )
        else:
            return []

    def theta_estimate(self, i, alpha=0.05):
        """Gets the posterior confidence interval and mean for a parameter.

        Args:
            i: The index of the parameter.
            alpha: The significance level for the confidence interval.

        Returns:
            A tuple containing the lower bound, mean, and upper bound of the
            parameter estimate.
        """

        if j < len(self.priors):
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


#     def efficacy_effect(self, j, alpha=0.05):
#         """ Get confidence interval and mean estimate of the effect on efficacy, expressed as odds-ratios.

#         Use:
#         - j=0, to get treatment effect of the intercept variable
#         - j=1, to get treatment effect of the pre-treated status variable
#         - j=2, to get treatment effect of the mutation status variable

#         """

#         if j==0:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,1])
#             return np.exp([self._pds.quantile(1, alpha/2), expected_log_or, self._pds.quantile(1, 1-alpha/2)])
#         elif j==1:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,2])
#             return np.exp([self._pds.quantile(2, alpha/2), expected_log_or, self._pds.quantile(2, 1-alpha/2)])
#         elif j==2:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,3])
#             return np.exp([self._pds.quantile(3, alpha/2), expected_log_or, self._pds.quantile(3, 1-alpha/2)])
#         else:
#             return (0,0,0)

#     def toxicity_effect(self, j=0, alpha=0.05):
#         """ Get confidence interval and mean estimate of the effect on toxicity, expressed as odds-ratios.

#         Use:
#         - j=0, to get effect on toxicity of the intercept variable

#         """

#         if j==0:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,0])
#             return np.exp([self._pds.quantile(0, alpha/2), expected_log_or, self._pds.quantile(0, 1-alpha/2)])
#         else:
#             return (0,0,0)

#     def correlation_effect(self, alpha=0.05):
#         """ Get confidence interval and mean estimate of the correlation between efficacy and toxicity. """
#         expected_psi = self._pds.expectation(self._pds._samp[:,4])
#         psi_levels = np.array([self._pds.quantile(4, alpha/2), expected_psi, self._pds.quantile(4, 1-alpha/2)])
#         return (np.exp(psi_levels) - 1) / (np.exp(psi_levels) + 1)
