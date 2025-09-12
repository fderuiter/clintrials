"""
Base classes and utilities for efficacy-toxicity dose-finding trials.
"""

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

logger = logging.getLogger(__name__)


class EfficacyToxicityDoseFindingTrial(metaclass=abc.ABCMeta):
    """An abstract base class for dose-finding trials that jointly monitor
    toxicity and efficacy.
    """

    def __init__(self, first_dose, num_doses, max_size):
        """Initializes an EfficacyToxicityDoseFindingTrial object.

        Args:
            first_dose (int): The starting dose level (1-based).
            num_doses (int): The total number of dose levels.
            max_size (int): The maximum number of patients in the trial.

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
            int: The trial status code.
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
            int: The number of dose levels.
        """
        return self.num_doses

    def dose_levels(self):
        """Gets a list of the dose levels (1-based indices).

        Returns:
            list[int]: A list of dose levels.
        """
        return list(range(1, self.num_doses + 1))

    def first_dose(self):
        """Gets the starting dose level.

        Returns:
            int: The first dose level.
        """
        return self._first_dose

    def size(self):
        """Gets the current number of treated patients.

        Returns:
            int: The number of patients treated so far.
        """
        return len(self._doses)

    def max_size(self):
        """Gets the maximum number of patients for the trial.

        Returns:
            int: The maximum trial size.
        """
        return self._max_size

    def doses(self):
        """Gets the list of doses given to patients.

        Returns:
            list[int]: A list of dose levels.
        """
        return self._doses

    def toxicities(self):
        """Gets the list of observed toxicities.

        Returns:
            list[int]: A list of toxicity outcomes (1 for toxicity, 0 for no
                toxicity).
        """
        return self._toxicities

    def efficacies(self):
        """Gets the list of observed efficacies.

        Returns:
            list[int]: A list of efficacy outcomes (1 for efficacy, 0 for no
                efficacy).
        """
        return self._efficacies

    def treated_at_dose(self, dose):
        """Gets the number of patients treated at a specific dose level.

        Args:
            dose (int): The 1-based dose level.

        Returns:
            int: The number of patients treated at the given dose.
        """
        return sum(np.array(self._doses) == dose)

    def toxicities_at_dose(self, dose):
        """Gets the number of toxicities observed at a specific dose level.

        Args:
            dose (int): The 1-based dose level.

        Returns:
            int: The number of toxicities at the given dose.
        """
        return sum([t for d, t in zip(self.doses(), self.toxicities()) if d == dose])

    def efficacies_at_dose(self, dose):
        """Gets the number of efficacies observed at a specific dose level.

        Args:
            dose (int): The 1-based dose level.

        Returns:
            int: The number of efficacies at the given dose.
        """
        return sum([e for d, e in zip(self.doses(), self.efficacies()) if d == dose])

    def maximum_dose_given(self):
        """Gets the maximum dose level administered so far.

        Returns:
            int | None: The maximum dose level, or `None` if no patients
                have been treated.
        """
        if len(self._doses) > 0:
            return max(self._doses)
        else:
            return None

    def minimum_dose_given(self):
        """Gets the minimum dose level administered so far.

        Returns:
            int | None: The minimum dose level, or `None` if no patients
                have been treated.
        """
        if len(self._doses) > 0:
            return min(self._doses)
        else:
            return None

    def tabulate(self):
        """Generates a summary table of the trial data.

        Returns:
            pandas.DataFrame: A DataFrame with the summary of patients,
                efficacies, and toxicities for each dose level.
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
        """Sets the next dose to be administered."""
        self._next_dose = dose

    def next_dose(self):
        """Gets the next dose to be administered.

        Returns:
            int: The next dose level.
        """
        return self._next_dose

    def update(self, cases, **kwargs):
        """Updates the trial with a list of new cases.

        Args:
            cases (list[tuple[int, int, int]]): A list of cases, where each
                case is a tuple of (dose, toxicity, efficacy).
            **kwargs: Additional keyword arguments for the dose calculation.

        Returns:
            int: The next recommended dose level.
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
        """Gets the current set of admissible doses.

        Returns:
            list[int]: The list of admissible dose levels.
        """
        return self._admissable_set

    def dose_admissability(self):
        """Gets a boolean array indicating the admissibility of each dose.

        Returns:
            numpy.ndarray: A boolean array where `True` indicates that the
                dose is admissible.
        """
        return np.array([(x in self._admissable_set) for x in self.dose_levels()])

    def observed_toxicity_rates(self):
        """Gets the observed toxicity rate for each dose level.

        Returns:
            numpy.ndarray: An array of observed toxicity rates.
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
        """Gets the observed efficacy rate for each dose level.

        Returns:
            numpy.ndarray: An array of observed efficacy rates.
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

        Args:
            prob_tox (list[float]): A list of toxicity probabilities.
            prob_eff (list[float]): A list of efficacy probabilities.

        Returns:
            int: The optimal 1-based dose level.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __reset(self):
        """Performs implementation-specific reset operations."""
        return

    @abc.abstractmethod
    def has_more(self):
        """Checks if the trial is ongoing.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return (self.size() < self.max_size()) and (self._status >= 0)

    @abc.abstractmethod
    def __calculate_next_dose(self, **kwargs):
        """Calculates the next dose to be administered.

        Subclasses should override this method.

        Returns:
            int: The next recommended dose level.
        """
        return -1


def _efftox_patient_outcome_to_label(po):
    """Converts a patient outcome tuple to a string label.

    Args:
        po (tuple[int, int]): A tuple representing the patient outcome,
            where the first element is toxicity (1 or 0) and the second is
            efficacy (1 or 0).

    Returns:
        str: A string label for the outcome (e.g., "Neither", "Toxicity",
            "Efficacy", "Both").
    """
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
    """Simulates a single efficacy-toxicity dose-finding trial.

    Args:
        design (EfficacyToxicityDoseFindingTrial): The trial design to use.
        true_toxicities (list[float]): The true toxicity rates for each dose.
        true_efficacies (list[float]): The true efficacy rates for each dose.
        tox_eff_odds_ratio (float, optional): The odds ratio for the
            association between toxicity and efficacy. Defaults to 1.0.
        tolerances (numpy.ndarray, optional): An array of uniform random
            numbers for simulating patient outcomes. If `None`, random
            numbers are generated. Defaults to `None`.
        cohort_size (int, optional): The number of patients per cohort.
            Defaults to 1.
        conduct_trial (bool, optional): If `True`, conducts the trial
            cohort-by-cohort. Defaults to `True`.
        calculate_optimal_decision (bool, optional): If `True`, calculates
            the optimal dose decision. Defaults to `True`.

    Returns:
        collections.OrderedDict: A dictionary containing the simulation report.

    Raises:
        ValueError: If the lengths of `true_efficacies`, `true_toxicities`,
            and the number of doses in the design do not match, or if
            `tolerances` has incorrect dimensions.
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
    """Simulates a single efficacy-toxicity dose-finding trial.

    This function is a wrapper around `_simulate_trial` that performs input
    validation and generates random tolerances if not provided.

    Args:
        design (EfficacyToxicityDoseFindingTrial): The trial design to use.
        true_toxicities (list[float]): The true toxicity rates for each dose.
        true_efficacies (list[float]): The true efficacy rates for each dose.
        tox_eff_odds_ratio (float, optional): The odds ratio for the
            association between toxicity and efficacy. Defaults to 1.0.
        tolerances (numpy.ndarray, optional): An array of uniform random
            numbers for simulating patient outcomes. If `None`, random
            numbers are generated. Defaults to `None`.
        cohort_size (int, optional): The number of patients per cohort.
            Defaults to 1.
        conduct_trial (bool, optional): If `True`, conducts the trial
            cohort-by-cohort. Defaults to `True`.
        calculate_optimal_decision (bool, optional): If `True`, calculates
            the optimal dose decision. Defaults to `True`.

    Returns:
        collections.OrderedDict: A dictionary containing the simulation report.
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

    This method allows for the comparison of different designs on the same set
    of simulated patient outcomes.

    Args:
        design_map (dict[str, EfficacyToxicityDoseFindingTrial]): A dictionary
            mapping design labels to trial design objects.
        true_toxicities (list[float]): The true toxicity rates for each dose.
        true_efficacies (list[float]): The true efficacy rates for each dose.
        tox_eff_odds_ratio (float, optional): The odds ratio for the
            association between toxicity and efficacy. Defaults to 1.0.
        tolerances (numpy.ndarray, optional): An array of uniform random
            numbers for simulating patient outcomes. If `None`, random
            numbers are generated. Defaults to `None`.
        cohort_size (int, optional): The number of patients per cohort.
            Defaults to 1.
        conduct_trial (bool, optional): If `True`, conducts the trial
            cohort-by-cohort. Defaults to `True`.
        calculate_optimal_decision (bool, optional): If `True`, calculates
            the optimal dose decision. Defaults to `True`.

    Returns:
        collections.OrderedDict: A dictionary of simulation reports, with
            keys corresponding to the design labels.
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
    """Calculates the dose-transition pathways for an efficacy-toxicity design.

    Args:
        trial (EfficacyToxicityDoseFindingTrial): The trial design object.
        next_dose (int): The dose to be given to the next cohort.
        cohort_sizes (list[int]): A list of future cohort sizes.
        cohort_number (int, optional): The starting cohort number.
            Defaults to 1.
        cases_already_observed (list, optional): A list of cases that have
            already been observed. Defaults to an empty list.
        custom_output_func (callable, optional): A function that takes the
            trial object and returns a dictionary of extra output. Defaults
            to `None`.
        verbose (bool, optional): If `True`, prints progress information.
            Defaults to `False`.
        **kwargs: Additional keyword arguments to pass to the `trial.update`
            method.

    Returns:
        dict: A nested dictionary representing the dose-transition pathways.
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
    """Constructs a string representation of a dose-transition path.

    Args:
        x (dict): A dictionary representing a single step in the DTP.
        dose_label_func (callable, optional): A function to format the dose
            label. Defaults to `str`.

    Returns:
        str: A string representation of the path.
    """
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    path = [x[z] for z in sorted([z for z in x.keys() if "Pat" in z])]
    path = [z[0] for z in path]
    path = "".join(path)
    path = dose_label_func(x["DoseGiven"]) + path
    return path


def print_dtps(dtps, indent=0, dose_label_func=None):
    """Prints the dose-transition pathways.

    Args:
        dtps (dict): A nested dictionary of DTPs.
        indent (int, optional): The indentation level. Defaults to 0.
        dose_label_func (callable, optional): A function to format the dose
            label. Defaults to `str`.
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
    """Prints the dose-transition pathways with verbose information.

    Args:
        dtps (dict): A nested dictionary of DTPs.
        indent (int, optional): The indentation level. Defaults to 0.
        dose_label_func (callable, optional): A function to format the dose
            label. Defaults to `str`.
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
