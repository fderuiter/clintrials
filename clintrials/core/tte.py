"""
Time-to-event trial designs.
"""

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


from collections import OrderedDict

import numpy as np
from scipy.stats import expon, invgamma, poisson

from clintrials.utils import atomic_to_json, iterable_to_json


class BayesianTimeToEvent:
    """A Bayesian design for time-to-event endpoints.

    This class implements a simple adaptive Bayesian design for time-to-event
    endpoints. It assumes exponentially distributed event times and an
    inverse-gamma prior on the median survival time.

    See Thall, P.F., Wooten, L.H., & Tannir, N.M. (2005) - "Monitoring Event
    Times in Early Phase Clinical Trials: Some Practical Issues" for details.
    """

    def __init__(self, alpha_prior, beta_prior):
        """Initializes a BayesianTimeToEvent object.

        Args:
            alpha_prior (float): The alpha parameter of the inverse-gamma prior
                on the median time-to-event.
            beta_prior (float): The beta parameter of the inverse-gamma prior on
                the median time-to-event.
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._times_to_event = []
        self._recruitment_times = []

    def event_times(self):
        """Gets the list of event times.

        Returns:
            list[float]: A list of event times in the order they were provided.
        """
        return self._times_to_event

    def recruitment_times(self):
        """Gets the list of recruitment times.

        Returns:
            list[float]: A list of recruitment times in the order they were
                provided.
        """
        return self._recruitment_times

    def update(self, cases):
        """Updates the trial with new patient cases.

        Args:
            cases (list[tuple[float, float]]): A list of cases, where each case
                is a tuple of (event_time, recruitment_time).
        """
        for event_time, recruitment_time in cases:
            self._times_to_event.append(event_time)
            self._recruitment_times.append(recruitment_time)

    def test(self, time, cutoff, probability, less_than=True):
        """Tests the posterior belief about the median time-to-event.

        This method tests whether the median time-to-event is less than or
        greater than a certain cutoff value, based on the posterior
        distribution.

        Args:
            time (float): The time at which to perform the test.
            cutoff (float): The critical value to test the median time against.
            probability (float): The required posterior certainty to declare
                significance.
            less_than (bool, optional): If `True`, tests if the parameter is
                less than the cutoff. If `False`, tests if it is greater.
                Defaults to `True`.

        Returns:
            collections.OrderedDict: A dictionary containing the test results.
        """
        event_time = np.array(self._times_to_event)
        recruit_time = np.array(self._recruitment_times)

        # Filter to just patients who are registered by time
        registered_patients = recruit_time <= time
        has_failed = (
            time - recruit_time[registered_patients] > event_time[registered_patients]
        )
        survival_time = np.array(
            [
                min(x, y)
                for (x, y) in zip(
                    time - recruit_time[registered_patients],
                    event_time[registered_patients],
                )
            ]
        )
        # Update posterior beliefs for mu_E
        alpha_post = self.alpha_prior + sum(has_failed)
        beta_post = self.beta_prior + np.log(2) * sum(survival_time)
        mu_post = beta_post / (alpha_post - 1)

        # Run test:
        test_probability = (
            invgamma.cdf(cutoff, a=alpha_post, scale=beta_post)
            if less_than
            else 1 - invgamma.cdf(cutoff, a=alpha_post, scale=beta_post)
        )
        stop_trial = (
            test_probability > probability
            if less_than
            else test_probability < probability
        )

        test_report = OrderedDict()
        test_report["Time"] = time
        test_report["Patients"] = sum(registered_patients)
        test_report["Events"] = sum(has_failed)
        test_report["TotalEventTime"] = sum(survival_time)
        test_report["AlphaPosterior"] = alpha_post
        test_report["BetaPosterior"] = beta_post
        test_report["MeanEventTimePosterior"] = mu_post
        test_report["MedianEventTimePosterior"] = mu_post * np.log(2)
        test_report["Cutoff"] = cutoff
        test_report["Certainty"] = probability
        test_report["Probability"] = test_probability
        test_report["LessThan"] = atomic_to_json(less_than)
        test_report["Stop"] = atomic_to_json(stop_trial)
        return test_report


def matrix_cohort_analysis(
    n_simulations,
    n_patients,
    true_median,
    alpha_prior,
    beta_prior,
    lower_cutoff,
    upper_cutoff,
    interim_certainty,
    final_certainty,
    interim_analysis_after_patients,
    interim_analysis_time_delta,
    final_analysis_time_delta,
    recruitment_stream,
):
    """Simulates time-to-event outcomes for a cohort.

    This function simulates a time-to-event trial based on the design of the
    National Lung Matrix trial.

    Args:
        n_simulations (int): The number of simulations to run.
        n_patients (int): The number of patients in the cohort.
        true_median (float): The true median time-to-event.
        alpha_prior (float): The alpha parameter of the inverse-gamma prior.
        beta_prior (float): The beta parameter of the inverse-gamma prior.
        lower_cutoff (float): The lower cutoff for the median time-to-event
            at interim analyses.
        upper_cutoff (float): The upper cutoff for the median time-to-event
            at the final analysis.
        interim_certainty (float): The required posterior certainty for
            stopping at interim analyses.
        final_certainty (float): The required posterior certainty for the
            final analysis.
        interim_analysis_after_patients (list[int]): A list of patient counts
            after which to perform interim analyses.
        interim_analysis_time_delta (float): The time delta to add to the
            recruitment time for interim analyses.
        final_analysis_time_delta (float): The time delta to add to the
            last recruitment time for the final analysis.
        recruitment_stream (clintrials.core.recruitment.RecruitmentStream):
            The recruitment stream to use for simulating patient arrival.

    Returns:
        list or dict: A list of simulation reports, or a single report if
            `n_simulations` is 1.
    """
    reports = []
    for i in range(n_simulations):
        trial = BayesianTimeToEvent(alpha_prior, beta_prior)
        recruitment_stream.reset()
        # recruitment_times = np.arange(1, n_patients+1) / recruitment
        recruitment_times = np.array(
            [recruitment_stream.next() for i in range(n_patients)]
        )
        true_mean = true_median / np.log(2)
        event_times = expon(scale=true_mean).rvs(
            n_patients
        )  # Exponential survival times
        cases = [(x, y) for (x, y) in zip(event_times, recruitment_times)]
        trial.update(cases)
        interim_analysis_times = list(
            {
                recruitment_times[x - 1] + interim_analysis_time_delta
                for x in interim_analysis_after_patients
                if x < n_patients
            }
        )

        trial_report = OrderedDict()
        # Call parameters
        trial_report["MaxPatients"] = n_patients
        trial_report["TrueMedianEventTime"] = true_median
        trial_report["PriorAlpha"] = alpha_prior
        trial_report["PriorBeta"] = beta_prior
        trial_report["LowerCutoff"] = lower_cutoff
        trial_report["UpperCutoff"] = upper_cutoff
        trial_report["InterimCertainty"] = interim_certainty
        trial_report["FinalCertainty"] = final_certainty
        trial_report["InterimAnalysisAfterPatients"] = interim_analysis_after_patients
        trial_report["InterimAnalysisTimeDelta"] = interim_analysis_time_delta
        trial_report["FinalAnalysisTimeDelta"] = final_analysis_time_delta
        # trial_report['Recruitment'] = recruitment
        # Simulated patient outcomes
        trial_report["RecruitmentTimes"] = iterable_to_json(recruitment_times)
        trial_report["EventTimes"] = iterable_to_json(event_times)
        trial_report["InterimAnalyses"] = []
        # Interim analyses
        for time in interim_analysis_times:
            interim_outcome = trial.test(
                time, lower_cutoff, interim_certainty, less_than=True
            )
            trial_report["InterimAnalyses"].append(interim_outcome)
            stop_trial = interim_outcome["Stop"]
            if stop_trial:
                trial_report["Decision"] = "StopAtInterim"
                trial_report["FinalAnalysis"] = interim_outcome
                trial_report["FinalPatients"] = interim_outcome["Patients"]
                trial_report["FinalEvents"] = interim_outcome["Events"]
                trial_report["FinalTotalEventTime"] = interim_outcome["TotalEventTime"]
                return trial_report
        # Final analysis
        final_analysis_time = max(recruitment_times) + final_analysis_time_delta
        final_outcome = trial.test(
            final_analysis_time, upper_cutoff, final_certainty, less_than=False
        )
        trial_report["FinalAnalysis"] = final_outcome
        stop_trial = final_outcome["Stop"]
        decision = "StopAtFinal" if stop_trial else "GoAtFinal"
        trial_report["Decision"] = decision
        trial_report["FinalPatients"] = final_outcome["Patients"]
        trial_report["FinalEvents"] = final_outcome["Events"]
        trial_report["FinalTotalEventTime"] = final_outcome["TotalEventTime"]
        reports.append(trial_report)

    if n_simulations == 1:
        return reports[0]
    else:
        return reports
