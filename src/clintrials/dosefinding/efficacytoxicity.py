__author__ = "brockk"


import abc
import logging
from collections import OrderedDict
from itertools import combinations_with_replacement, product

import numpy as np

from clintrials.utils import (
    atomic_to_json,
    correlated_binary_outcomes_from_uniforms,
    iterable_to_json,
    to_1d_list,
)

# from clintrials.simulation import filter_sims


logger = logging.getLogger(__name__)


# Joint Phase I/II, Assessing efficacy and toxicity
class EfficacyToxicityDoseFindingTrial(metaclass=abc.ABCMeta):
    """A base class for dose-finding trials that jointly monitor toxicity and efficacy.

    This class provides a common interface and functionality for efficacy-toxicity
    dose-finding trials. Subclasses should implement the abstract methods.

    Attributes:
        num_doses: The number of dose levels under investigation.
    """

    def __init__(self, first_dose, num_doses, max_size):
        """Initializes the EfficacyToxicityDoseFindingTrial.

        Args:
            first_dose: The starting dose level (1-based).
            num_doses: The total number of dose levels.
            max_size: The maximum number of patients in the trial.

        Raises:
            ValueError: If `first_dose` is greater than `num_doses`.
        """
        if first_dose > num_doses:
            raise ValueError("First dose must be no greater than number of doses.")

        self._first_dose = first_dose
        self.num_doses = num_doses
        self._max_size = max_size

        # Reset
        self._doses = []
        self._toxicities = []
        self._efficacies = []
        self._next_dose = self._first_dose
        self._status = 0
        self._admissable_set = []

    def status(self):
        """Gets the current status of the trial.

        Returns:
            The current trial status.
        """
        return self._status

    def reset(self):
        """Resets the trial to its initial state."""
        self._doses = []
        self._toxicities = []
        self._efficacies = []
        self._next_dose = self._first_dose
        self._status = 0
        self.__reset()

    def number_of_doses(self):
        """Gets the number of dose levels under investigation.

        Returns:
            The number of dose levels.
        """
        return self.num_doses

    def dose_levels(self):
        """Gets a list of all dose levels.

        Returns:
            A list of dose levels (1-based).
        """
        return range(1, self.num_doses + 1)

    def first_dose(self):
        """Gets the starting dose level.

        Returns:
            The starting dose level.
        """
        return self._first_dose

    def size(self):
        """Gets the number of patients treated so far.

        Returns:
            The number of patients treated.
        """
        return len(self._doses)

    def max_size(self):
        """Gets the maximum number of patients for the trial.

        Returns:
            The maximum number of patients.
        """
        return self._max_size

    def doses(self):
        """Gets the list of doses given to patients.

        Returns:
            A list of doses.
        """
        return self._doses

    def toxicities(self):
        """Gets the list of observed toxicities.

        Returns:
            A list of toxicities (1 for toxic, 0 for not).
        """
        return self._toxicities

    def efficacies(self):
        """Gets the list of observed efficacies.

        Returns:
            A list of efficacies (1 for effective, 0 for not).
        """
        return self._efficacies

    def treated_at_dose(self, dose):
        """Gets the number of patients treated at a specific dose level.

        Args:
            dose: The dose level (1-based).

        Returns:
            The number of patients treated at the given dose.
        """
        return sum(np.array(self._doses) == dose)

    def toxicities_at_dose(self, dose):
        """Gets the number of toxicities observed at a specific dose level.

        Args:
            dose: The dose level (1-based).

        Returns:
            The number of toxicities at the given dose.
        """
        return sum([t for d, t in zip(self.doses(), self.toxicities()) if d == dose])

    def efficacies_at_dose(self, dose):
        """Gets the number of efficacies observed at a specific dose level.

        Args:
            dose: The dose level (1-based).

        Returns:
            The number of efficacies at the given dose.
        """
        return sum([e for d, e in zip(self.doses(), self.efficacies()) if d == dose])

    def maximum_dose_given(self):
        """Gets the maximum dose level administered so far.

        Returns:
            The maximum dose level, or None if no doses have been given.
        """
        if len(self._doses) > 0:
            return max(self._doses)
        else:
            return None

    def minimum_dose_given(self):
        """Gets the minimum dose level administered so far.

        Returns:
            The minimum dose level, or None if no doses have been given.
        """
        if len(self._doses) > 0:
            return min(self._doses)
        else:
            return None

    def tabulate(self):
        """Creates a summary table of the trial results.

        Returns:
            A pandas DataFrame summarizing the trial results.
        """
        import pandas as pd

        tab_data = OrderedDict()
        treated_at_dose = [self.treated_at_dose(d) for d in self.dose_levels()]
        eff_at_dose = [self.efficacies_at_dose(d) for d in self.dose_levels()]
        tox_at_dose = [self.toxicities_at_dose(d) for d in self.dose_levels()]
        tab_data["Dose"] = self.dose_levels()
        tab_data["N"] = treated_at_dose
        tab_data["Efficacies"] = eff_at_dose
        tab_data["Toxicities"] = tox_at_dose
        df = pd.DataFrame(tab_data)
        df["EffRate"] = np.where(df.N > 0, df.Efficacies / df.N, np.nan)
        df["ToxRate"] = np.where(df.N > 0, df.Toxicities / df.N, np.nan)
        return df

    def set_next_dose(self, dose):
        """Sets the next dose to be administered.

        Args:
            dose: The next dose level.
        """
        self._next_dose = dose

    def next_dose(self):
        """Gets the next dose to be administered.

        Returns:
            The next dose level.
        """
        return self._next_dose

    def update(self, cases, **kwargs):
        """Updates the trial with a list of new cases.

        Args:
            cases: A list of 3-tuples, where each tuple is
                (dose, toxicity, efficacy).
            **kwargs: Additional keyword arguments for the dose calculation.

        Returns:
            The next recommended dose.
        """

        if len(cases) > 0:
            for dose, tox, eff in cases:
                self._doses.append(dose)
                self._toxicities.append(tox)
                self._efficacies.append(eff)

            self._next_dose = self.__calculate_next_dose(**kwargs)
        else:
            logging.warning("Cannot update design with no cases")

        return self._next_dose

    def admissable_set(self):
        """Gets the set of admissible doses.

        Returns:
            A list of admissible doses.
        """
        return self._admissable_set

    def dose_admissability(self):
        """Gets a boolean array indicating the admissibility of each dose.

        Returns:
            A numpy array of booleans.
        """
        return np.array([(x in self._admissable_set) for x in self.dose_levels()])

    def observed_toxicity_rates(self):
        """Gets the observed toxicity rate for each dose.

        Returns:
            A numpy array of observed toxicity rates.
        """
        tox_rates = []
        for d in range(1, self.num_doses + 1):
            num_treated = self.treated_at_dose(d)
            if num_treated:
                num_toxes = self.toxicities_at_dose(d)
                tox_rates.append(1.0 * num_toxes / num_treated)
            else:
                tox_rates.append(np.nan)
        return np.array(tox_rates)

    def observed_efficacy_rates(self):
        """Gets the observed efficacy rate for each dose.

        Returns:
            A numpy array of observed efficacy rates.
        """
        eff_rates = []
        for d in range(1, self.num_doses + 1):
            num_treated = self.treated_at_dose(d)
            if num_treated:
                num_responses = self.efficacies_at_dose(d)
                eff_rates.append(1.0 * num_responses / num_treated)
            else:
                eff_rates.append(np.nan)
        return np.array(eff_rates)

    def optimal_decision(self, prob_tox, prob_eff):
        """Gets the optimal dose choice for given toxicity and efficacy curves.

        This method determines the optimal dose based on given toxicity and
        efficacy curves, as described by Ken Cheung (2014).

        Args:
            prob_tox: A list of toxicity probabilities for each dose.
            prob_eff: A list of efficacy probabilities for each dose.

        Returns:
            The optimal dose level (1-based).
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def __reset(self):
        """Opportunity to run implementation-specific reset operations."""
        return

    @abc.abstractmethod
    def has_more(self):
        """Is the trial ongoing?"""
        return (self.size() < self.max_size()) and (self._status >= 0)

    @abc.abstractmethod
    def __calculate_next_dose(self, **kwargs):
        """Subclasses should override this method and return the desired next dose."""
        return -1  # Default implementation


def _efftox_patient_outcome_to_label(po):
    """Converts (0,0) to Neither; (1,0) to Toxicity, (0,1) to Efficacy, (1,1) to Both"""
    if po == (0, 0):
        return "Neither"
    elif po == (1, 0):
        return "Toxicity"
    elif po == (0, 1):
        return "Efficacy"
    elif po == (1, 1):
        return "Both"
    else:
        return "Error"


def _simulate_trial(
    design,
    true_toxicities,
    true_efficacies,
    tox_eff_odds_ratio=1.0,
    tolerances=None,
    cohort_size=1,
    conduct_trial=1,
    calculate_optimal_decision=1,
):
    """Simulate a dose finding trial based on efficacy and toxicity, like EffTox, etc.

    :param design: the design with which to simulate a dose-finding trial.
    :type design: clintrials.dosefinding.EfficacyToxicityDoseFindingTrial
    :param true_toxicities: list of the true toxicity rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_toxicities: list
    :param true_efficacies: list of the true efficacy rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_efficacies: list
    :param tox_eff_odds_ratio: odds ratio of toxicity and efficacy events. Use 1. for no association
    :type tox_eff_odds_ratio: float
    :param tolerances: optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    :type tolerances: numpy.array
    :param cohort_size: to add several patients at a dose at once
    :type cohort_size: int
    :param conduct_trial: True to conduct cohort-by-cohort dosing using the trial design; False to suppress
    :type conduct_trial: bool
    :param calculate_optimal_decision: True to calculate the optimal dose; False to suppress
    :type calculate_optimal_decision: bool

    :return: report of the simulation outcome as a JSON-able dict
    :rtype: dict

    """

    correlated_outcomes = tox_eff_odds_ratio < 1.0 or tox_eff_odds_ratio > 1.0

    # Simulate trial
    if conduct_trial:
        i = 0
        design.reset()
        dose_level = design.next_dose()
        while i <= design.max_size() and design.has_more():
            u = (true_toxicities[dose_level - 1], true_efficacies[dose_level - 1])
            if correlated_outcomes:
                # Where outcomes are associated, simulated outcomes must reflect the association.
                # There is a special method for that:
                events = correlated_binary_outcomes_from_uniforms(
                    tolerances[i : i + cohort_size,], u, psi=tox_eff_odds_ratio
                ).astype(int)
            else:
                # Outcomes are not associated. Simply use first two columns of tolerances as
                # uniformally-distributed thresholds for tox and eff. The third col is ignored.
                events = (tolerances[i : i + cohort_size, 0:2] < u).astype(int)
            cases = np.column_stack(([dose_level] * cohort_size, events))
            dose_level = design.update(cases)
            i += cohort_size

    # Report findings
    report = OrderedDict()
    # report['TrueToxicities'] = iterable_to_json(true_toxicities)
    # report['TrueEfficacies'] = iterable_to_json(true_efficacies)
    # Do not parrot back parameters

    if conduct_trial:
        report["RecommendedDose"] = atomic_to_json(design.next_dose())
        report["TrialStatus"] = atomic_to_json(design.status())
        report["Doses"] = iterable_to_json(design.doses())
        report["Toxicities"] = iterable_to_json(design.toxicities())
        report["Efficacies"] = iterable_to_json(design.efficacies())
    # Optimal decision, given these specific patient tolerances
    if calculate_optimal_decision:
        try:
            if correlated_outcomes:
                tox_eff_hat = np.array(
                    [
                        correlated_binary_outcomes_from_uniforms(
                            tolerances, v, psi=tox_eff_odds_ratio
                        ).mean(axis=0)
                        for v in zip(true_toxicities, true_efficacies)
                    ]
                )
                tox_hat, eff_hat = tox_eff_hat[:, 0], tox_eff_hat[:, 1]
            else:
                had_tox = lambda x: x < np.array(true_toxicities)
                tox_horizons = np.array([had_tox(x) for x in tolerances[:, 0]])
                tox_hat = tox_horizons.mean(axis=0)
                had_eff = lambda x: x < np.array(true_efficacies)
                eff_horizons = np.array([had_eff(x) for x in tolerances[:, 1]])
                eff_hat = eff_horizons.mean(axis=0)

            optimal_allocation = design.optimal_decision(tox_hat, eff_hat)
            report["FullyInformedToxicityCurve"] = iterable_to_json(
                np.round(tox_hat, 4)
            )
            report["FullyInformedEfficacyCurve"] = iterable_to_json(
                np.round(eff_hat, 4)
            )
            report["OptimalAllocation"] = atomic_to_json(optimal_allocation)
        except NotImplementedError:
            pass

    return report


# Alias
_simulate_eff_tox_trial = _simulate_trial


def simulate_trial(
    design,
    true_toxicities,
    true_efficacies,
    tox_eff_odds_ratio=1.0,
    tolerances=None,
    cohort_size=1,
    conduct_trial=1,
    calculate_optimal_decision=1,
):
    """Simulates a dose-finding trial that monitors efficacy and toxicity.

    Args:
        design: The trial design to simulate.
        true_toxicities: A list of the true toxicity rates for each dose.
        true_efficacies: A list of the true efficacy rates for each dose.
        tox_eff_odds_ratio: The odds ratio of toxicity and efficacy events.
        tolerances: An optional array of uniforms for generating correlated
            outcomes. If None, random data is used.
        cohort_size: The number of patients to add at each dose.
        conduct_trial: If True, conduct cohort-by-cohort dosing.
        calculate_optimal_decision: If True, calculate the optimal dose.

    Returns:
        A dictionary containing the simulation report.

    Raises:
        ValueError: If the lengths of `true_efficacies`, `true_toxicities`,
            and the number of doses in the design are inconsistent.
    """

    # Validation and derivation of the inputs
    if len(true_efficacies) != len(true_toxicities):
        raise ValueError("true_efficacies and true_toxicities should be same length.")
    if len(true_toxicities) != design.number_of_doses():
        raise ValueError(
            "Length of true_toxicities and number of doses should be the same."
        )
    n_patients = design.max_size()
    if tolerances is not None:
        if tolerances.ndim != 2 or tolerances.shape[0] < n_patients:
            raise ValueError("tolerances should be an n_patients*3 array")
    else:
        tolerances = np.random.uniform(size=3 * n_patients).reshape(n_patients, 3)

    if tox_eff_odds_ratio != 1.0 and calculate_optimal_decision:
        logging.warning(
            "Patient outcomes are not sequential when toxicity and efficacy events are correlated. "
            + "E.g. toxicity at d_1 dose not necessarily imply toxicity at d_2. It is important "
            + "to appreciate this when calculating optimal decisions."
        )

    return _simulate_trial(
        design,
        true_toxicities,
        true_efficacies,
        tox_eff_odds_ratio,
        tolerances,
        cohort_size,
        conduct_trial,
        calculate_optimal_decision,
    )


# Alias
simulate_efficacy_toxicity_dose_finding_trial = simulate_trial


def simulate_efficacy_toxicity_dose_finding_trials(
    design_map,
    true_toxicities,
    true_efficacies,
    tox_eff_odds_ratio=1.0,
    tolerances=None,
    cohort_size=1,
    conduct_trial=1,
    calculate_optimal_decision=1,
):
    """Simulates multiple efficacy-toxicity dose-finding trials.

    This function allows for the comparison of different designs on a common
    set of patient outcomes.

    Args:
        design_map: A dictionary mapping design labels to trial design
            instances.
        true_toxicities: A list of the true toxicity rates for each dose.
        true_efficacies: A list of the true efficacy rates for each dose.
        tox_eff_odds_ratio: The odds ratio of toxicity and efficacy events.
        tolerances: An optional array of uniforms for generating correlated
            outcomes. If None, random data is used.
        cohort_size: The number of patients to add at each dose.
        conduct_trial: If True, conduct cohort-by-cohort dosing.
        calculate_optimal_decision: If True, calculate the optimal dose.

    Returns:
        A dictionary containing the simulation reports for each design.
    """

    max_size = max([design.max_size() for design in design_map.values()])
    if tolerances is not None:
        if tolerances.ndim != 2 or tolerances.shape[0] < max_size:
            raise ValueError("tolerances should be an max_size*3 array")
    else:
        tolerances = np.random.uniform(size=3 * max_size).reshape(max_size, 3)

    if tox_eff_odds_ratio != 1.0 and calculate_optimal_decision:
        logging.warning(
            "Patient outcomes are not sequential when toxicity and efficacy events are correlated. "
            + "E.g. toxicity at d_1 dose not necessarily imply toxicity at d_2. It is important "
            + "to appreciate this when calculating optimal decisions."
        )

    report = OrderedDict()
    # report['TrueToxicities'] = iterable_to_json(true_toxicities)
    # report['TrueEfficacies'] = iterable_to_json(true_efficacies)
    # Do not parrot back parameters

    for label, design in design_map.items():
        this_sim = _simulate_trial(
            design,
            true_toxicities,
            true_efficacies,
            tox_eff_odds_ratio,
            tolerances,
            cohort_size,
            conduct_trial,
            calculate_optimal_decision,
        )
        report[label] = this_sim

    return report


# Alias
simulate_trials = simulate_efficacy_toxicity_dose_finding_trials


def dose_transition_pathways(
    trial,
    next_dose,
    cohort_sizes,
    cohort_number=1,
    cases_already_observed=[],
    custom_output_func=None,
    verbose=False,
    **kwargs,
):
    """Calculates dose-transition pathways for an efficacy-toxicity design.

    Args:
        trial: The trial design to use.
        next_dose: The dose to be given to the next cohort.
        cohort_sizes: A list of future cohort sizes.
        cohort_number: The starting cohort number.
        cases_already_observed: A list of cases already observed.
        custom_output_func: An optional function to generate custom output.
        verbose: If True, print progress information.
        **kwargs: Extra keyword arguments for the `trial.update` method.

    Returns:
        A dictionary representing the dose-transition pathways.
    """

    if len(cohort_sizes) <= 0:
        return None
    else:
        cohort_size = cohort_sizes[0]
        patient_outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        cohort_outcomes = list(
            combinations_with_replacement(patient_outcomes, cohort_size)
        )
        path_outputs = []
        for i, path in enumerate(cohort_outcomes):
            # Invoke dose-decision
            cohort_cases = [(next_dose, x[0], x[1]) for x in path]
            cases = cases_already_observed + cohort_cases
            if verbose:
                logger.debug("Running %s", cases)
            trial.reset()
            obd = trial.update(cases, **kwargs)
            # Collect output
            bag_o_tricks = OrderedDict(
                [
                    (f"Pat{cohort_number}.{j+1}", _efftox_patient_outcome_to_label(po))
                    for (j, po) in enumerate(path)
                ]
            )
            bag_o_tricks.update(
                OrderedDict(
                    [
                        ("DoseGiven", atomic_to_json(next_dose)),
                        ("RecommendedDose", atomic_to_json(obd)),
                        ("CohortSize", cohort_size),
                        ("NumEff", sum([x[1] for x in path])),
                        ("NumTox", sum([x[0] for x in path])),
                    ]
                )
            )
            if custom_output_func:
                bag_o_tricks.update(custom_output_func(trial))

            # Recurse subsequent cohorts
            further_paths = dose_transition_pathways(
                trial,
                next_dose=obd,
                cohort_sizes=cohort_sizes[1:],
                cohort_number=cohort_number + 1,
                cases_already_observed=cases,
                custom_output_func=custom_output_func,
                verbose=verbose,
                **kwargs,
            )
            if further_paths:
                bag_o_tricks["Next"] = further_paths

            path_outputs.append(bag_o_tricks)

        return path_outputs


# Aliases
efftox_dose_transition_pathways = dose_transition_pathways
efficacy_toxicity_dose_transition_pathways = dose_transition_pathways


def get_path(x, dose_label_func=None):
    """Constructs a path string from a dose-transition pathway node.

    Args:
        x: A dictionary representing a node in the dose-transition pathway.
        dose_label_func: An optional function to format the dose label.

    Returns:
        A string representing the path.
    """
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    path = [x[z] for z in sorted([z for z in x.keys() if "Pat" in z])]
    path = [z[0] for z in path]
    path = "".join(path)
    path = dose_label_func(x["DoseGiven"]) + path
    return path


def print_dtps(dtps, indent=0, dose_label_func=None):
    """Prints a summary of the dose-transition pathways.

    Args:
        dtps: A list of dose-transition pathway nodes.
        indent: The indentation level for printing.
        dose_label_func: An optional function to format the dose label.
    """
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    for x in dtps:
        path = get_path(x, dose_label_func=dose_label_func)
        obd = x["RecommendedDose"]
        prob_sup = x["MinProbSuperiority"]

        if prob_sup < 0.6:
            template_txt = "\t" * indent + "{} -> Dose {}, Superiority={} * tentative *"
        else:
            template_txt = "\t" * indent + "{} -> Dose {}, Superiority={}"
        logger.info(
            template_txt.format(path, dose_label_func(obd), np.round(prob_sup, 2))
        )

        if "Next" in x:
            print_dtps(x["Next"], indent=indent + 1, dose_label_func=dose_label_func)


def print_dtps_verbose(dtps, indent=0, dose_label_func=None):
    """Prints a verbose summary of the dose-transition pathways.

    Args:
        dtps: A list of dose-transition pathway nodes.
        indent: The indentation level for printing.
        dose_label_func: An optional function to format the dose label.
    """
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    for x in dtps:
        path = get_path(x, dose_label_func=dose_label_func)
        obd = x["RecommendedDose"]
        prob_sup = x["MinProbSuperiority"]
        util = [x["Utility1"], x["Utility2"], x["Utility3"], x["Utility4"]]
        prob_acc_eff = [
            x["ProbAccEff1"],
            x["ProbAccEff2"],
            x["ProbAccEff3"],
            x["ProbAccEff4"],
        ]
        prob_acc_tox = [
            x["ProbAccTox1"],
            x["ProbAccTox2"],
            x["ProbAccTox3"],
            x["ProbAccTox4"],
        ]
        template_txt = (
            "\t" * indent
            + "{} -> Dose {}, Sup={}, Util={}, Pr(Acc Eff)={}, Pr(Acc Tox)={}"
        )
        logger.info(
            template_txt.format(
                path,
                dose_label_func(obd),
                np.round(prob_sup, 2),
                np.round(util, 2),
                np.round(prob_acc_eff, 2),
                np.round(prob_acc_tox, 2),
            )
        )

        if "Next" in x:
            print_dtps_verbose(
                x["Next"], indent=indent + 1, dose_label_func=dose_label_func
            )
