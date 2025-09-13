__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


__all__ = ["crm", "efftox", "efficacytoxicity", "wagestait", "watu"]


import abc
import copy
import logging
from collections import OrderedDict
from itertools import combinations_with_replacement, product

import numpy as np
from scipy.stats import uniform

from clintrials.utils import (
    atomic_to_json,
    correlated_binary_outcomes_from_uniforms,
    filter_list_of_dicts,
    iterable_to_json,
    to_1d_list,
)

logger = logging.getLogger(__name__)


class DoseFindingTrial(metaclass=abc.ABCMeta):
    """Base class for a dose-finding trial."""

    def __init__(self, first_dose, num_doses, max_size):
        """Initializes a DoseFindingTrial object.

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
        self._next_dose = self._first_dose
        self._status = 0

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
        return range(1, self.num_doses + 1)

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
            pandas.DataFrame: A DataFrame with the summary of patients and
                toxicities for each dose level.
        """
        import pandas as pd

        tab_data = OrderedDict()
        treated_at_dose = [self.treated_at_dose(d) for d in self.dose_levels()]
        tox_at_dose = [self.toxicities_at_dose(d) for d in self.dose_levels()]
        tab_data["Dose"] = self.dose_levels()
        tab_data["N"] = treated_at_dose
        tab_data["Toxicities"] = tox_at_dose
        df = pd.DataFrame(tab_data)
        df["ToxRate"] = np.where(df.N > 0, df.Toxicities / df.N, np.nan)
        return df

    def set_next_dose(self, dose):
        """Sets the next dose to be administered.

        Args:
            dose (int): The next dose level.
        """
        self._next_dose = dose

    def next_dose(self):
        """Gets the next dose to be administered.

        Returns:
            int: The next dose level.
        """
        return self._next_dose

    def update(self, cases):
        """Updates the trial with a list of new cases.

        Args:
            cases (list[tuple[int, int]]): A list of cases, where each case
                is a tuple of (dose, toxicity).

        Returns:
            int: The next recommended dose level.
        """

        for dose, tox in cases:
            self._doses.append(dose)
            self._toxicities.append(tox)

        self._next_dose = self.__calculate_next_dose()
        return self._next_dose

    def observed_toxicity_rates(self):
        """Gets the observed rate of toxicity at all doses.

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

    def optimal_decision(self, prob_tox):
        """Gets the optimal dose choice for a given dose-toxicity curve.

        Args:
            prob_tox (list[float]): A list of toxicity probabilities for each
                dose level.

        Returns:
            int: The optimal 1-based dose level.
        """
        raise NotImplementedError()

    def plot_outcomes(self, chart_title=None, use_ggplot=False):
        """Plots the outcomes of patients observed.

        Args:
            chart_title (str, optional): The title for the chart. Defaults to
                a descriptive title.
            use_ggplot (bool, optional): If `True`, uses ggplot for plotting.
                Otherwise, uses matplotlib. Defaults to `False`.

        Returns:
            A plot object.
        """
        if not chart_title:
            chart_title = "Each point represents a patient\nA circle indicates no toxicity, a cross toxicity"
            chart_title = chart_title + "\n"

        if use_ggplot:
            if self.size() > 0:
                import numpy as np
                import pandas as pd
                from ggplot import aes, geom_text, ggplot, ggtitle, ylim

                patient_number = range(1, self.size() + 1)
                symbol = np.where(self.toxicities(), "X", "O")
                data = pd.DataFrame(
                    {
                        "Patient number": patient_number,
                        "Dose level": self.doses(),
                        "DLT": self.toxicities(),
                        "Symbol": symbol,
                    }
                )

                p = (
                    ggplot(
                        data, aes(x="Patient number", y="Dose level", label="Symbol")
                    )
                    + ggtitle(chart_title)
                    + geom_text(aes(size=20, vjust=-0.07))
                    + ylim(1, 5)
                )
                return p
        else:
            if self.size() > 0:
                import matplotlib.pyplot as plt
                import numpy as np

                patient_number = np.arange(1, self.size() + 1)
                doses_given = np.array(self.doses())
                tox_loc = np.array(self.toxicities()).astype("bool")
                if sum(tox_loc):
                    plt.scatter(
                        patient_number[tox_loc],
                        doses_given[tox_loc],
                        marker="x",
                        s=300,
                        facecolors="none",
                        edgecolors="k",
                    )
                if sum(~tox_loc):
                    plt.scatter(
                        patient_number[~tox_loc],
                        doses_given[~tox_loc],
                        marker="o",
                        s=300,
                        facecolors="none",
                        edgecolors="k",
                    )

                plt.title(chart_title)
                plt.ylabel("Dose level")
                plt.xlabel("Patient number")
                plt.yticks(self.dose_levels())
                p = plt.gcf()
                phi = (np.sqrt(5) + 1) / 2.0
                p.set_size_inches(12, 12 / phi)
                # return p

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
    def __calculate_next_dose(self):
        """Calculates the next dose to be administered."""
        return -1  # Default implementation


class SimpleToxicityCountingDoseEscalationTrial(DoseFindingTrial):
    """A simple design that monotonically increases the dose until a
    certain number of toxicities are observed.
    """

    def __init__(self, first_dose, num_doses, max_size, max_toxicities=1):
        """Initializes a SimpleToxicityCountingDoseEscalationTrial object.

        Args:
            first_dose (int): The starting dose level (1-based).
            num_doses (int): The total number of dose levels.
            max_size (int): The maximum number of patients in the trial.
            max_toxicities (int, optional): The maximum number of toxicities
                allowed before stopping. Defaults to 1.
        """
        DoseFindingTrial.__init__(
            self, first_dose=first_dose, num_doses=num_doses, max_size=max_size
        )

        self.max_toxicities = max_toxicities
        # Reset
        self.max_dose_given = -1

    def _DoseFindingTrial__reset(self):
        self.max_dose_given = -1

    def _DoseFindingTrial__calculate_next_dose(self):
        if self.has_more():
            self._status = 1
            if len(self.doses()) > 0:
                return min(max(self.doses()) + 1, self.number_of_doses())
            else:
                return self._first_dose
        else:
            self._status = 100
            return max(self.doses())

    def has_more(self):
        """Checks if the trial is ongoing.

        The trial stops if the maximum number of patients is reached, the
        maximum number of toxicities is observed, or the highest dose level
        is reached.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return (
            DoseFindingTrial.has_more(self)
            and (sum(self.toxicities()) < self.max_toxicities)
            and self.maximum_dose_given() < self.number_of_doses()
        )


class ThreePlusThree(DoseFindingTrial):
    """An object-oriented implementation of the 3+3 trial design."""

    def __init__(self, num_doses):
        """Initializes a ThreePlusThree trial object.

        Args:
            num_doses (int): The total number of dose levels.
        """
        DoseFindingTrial.__init__(
            self, first_dose=1, num_doses=num_doses, max_size=6 * num_doses
        )

        self.num_doses = num_doses
        self.cohort_size = 3
        # Reset
        self._continue = True

    def _DoseFindingTrial__reset(self):
        self._continue = True

    def _DoseFindingTrial__calculate_next_dose(self):
        dose_indices = np.array(self._doses) == self._next_dose
        toxes_at_dose = sum(np.array(self._toxicities)[dose_indices])
        if sum(dose_indices) == 3:
            if toxes_at_dose == 0:
                if self._next_dose < self.num_doses:
                    # escalate
                    self._status = 1
                    self._next_dose += 1
                else:
                    # end trial
                    self._status = 100
                    self._continue = False
            elif toxes_at_dose == 1:
                # Do not escalate but continue trial
                self._status = 1
                pass
            else:
                # too many toxicities at this dose so de-escalate and end trial
                self._next_dose -= 1
                if self._next_dose > 0:
                    self._status = 100
                else:
                    self._status = -1
                self._continue = False
        elif sum(dose_indices) == 6:
            if toxes_at_dose <= 1:
                if self._next_dose < self.num_doses:
                    # escalate
                    self._status = 1
                    self._next_dose += 1
                else:
                    # end trial
                    self._status = 100
                    self._continue = False
            else:
                # too many toxicities at this dose so de-escalate and end trial
                self._next_dose -= 1
                if self._next_dose > 0:
                    self._status = 100
                else:
                    self._status = -1
                self._continue = False
        else:
            msg = "Doses in the 3+3 trial must be given in common batches of three."
            raise Exception(msg)

        return self._next_dose

    def has_more(self):
        """Checks if the trial is ongoing.

        The 3+3 trial stops when the MTD has been found.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return DoseFindingTrial.has_more(self) and self._continue


def simulate_dose_finding_trial(
    design,
    true_toxicities,
    tolerances=None,
    cohort_size=1,
    conduct_trial=True,
    calculate_optimal_decision=True,
):
    """Simulates a dose-finding trial.

    Args:
        design (DoseFindingTrial): The trial design to use.
        true_toxicities (list[float]): The true toxicity rates for each
            dose level.
        tolerances (list[float], optional): A list of uniform random
            numbers for simulating patient outcomes. Defaults to `None`.
        cohort_size (int, optional): The number of patients per cohort.
            Defaults to 1.
        conduct_trial (bool, optional): If `True`, conducts the trial
            cohort-by-cohort. Defaults to `True`.
        calculate_optimal_decision (bool, optional): If `True`, calculates
            the optimal dose decision. Defaults to `True`.

    Returns:
        collections.OrderedDict: A dictionary containing the simulation report.
    """
    # Validate inputs
    if tolerances is None:
        tolerances = uniform().rvs(design.max_size())
    else:
        if len(tolerances) < design.max_size():
            logging.warning(
                "You have provided fewer tolerances than maximum number of patients on trial. Beware errors!"
            )

    # Simulate trial
    if conduct_trial:
        i = 0
        design.reset()
        dose_level = design.next_dose()
        while i <= design.max_size() and design.has_more():
            tox = [
                1 if x < true_toxicities[dose_level - 1] else 0
                for x in tolerances[i : i + cohort_size]
            ]
            cases = zip([dose_level] * cohort_size, tox)
            dose_level = design.update(cases)
            i += cohort_size

    # Report findings
    report = OrderedDict()
    report["TrueToxicities"] = iterable_to_json(true_toxicities)
    if conduct_trial:
        report["RecommendedDose"] = atomic_to_json(design.next_dose())
        report["TrialStatus"] = atomic_to_json(design.status())
        report["Doses"] = iterable_to_json(design.doses())
        report["Toxicities"] = iterable_to_json(design.toxicities())
    # Optimal decision, given these specific patient tolerances
    if calculate_optimal_decision:
        try:
            had_tox = lambda x: x < np.array(true_toxicities)
            tox_horizons = np.array([had_tox(x) for x in tolerances])
            tox_hat = tox_horizons.mean(axis=0)

            optimal_allocation = design.optimal_decision(tox_hat)
            report["FullyInformedToxicityCurve"] = iterable_to_json(tox_hat)
            report["OptimalAllocation"] = atomic_to_json(optimal_allocation)
        except NotImplementedError:
            pass

    return report


def simulate_dose_finding_trials(
    design_map,
    true_toxicities,
    tolerances=None,
    cohort_size=1,
    conduct_trial=True,
    calculate_optimal_decision=True,
):
    """Simulates multiple toxicity-driven dose-finding trials from the same
    patient data.

    Args:
        design_map (dict[str, DoseFindingTrial]): A dictionary mapping
            design labels to trial design objects.
        true_toxicities (list[float]): The true toxicity rates for each
            dose level.
        tolerances (list[float], optional): A list of uniform random
            numbers for simulating patient outcomes. Defaults to `None`.
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
    if tolerances is None:
        tolerances = uniform().rvs(max_size)
    else:
        if len(tolerances) < max_size:
            logging.warning(
                "You have provided fewer tolerances than maximum number of patients on trial. Beware errors!"
            )

    report = OrderedDict()
    report["TrueToxicities"] = iterable_to_json(true_toxicities)
    for label, design in design_map.items():
        design_sim = simulate_dose_finding_trial(
            design,
            true_toxicities,
            tolerances=tolerances,
            cohort_size=cohort_size,
            conduct_trial=conduct_trial,
            calculate_optimal_decision=calculate_optimal_decision,
        )
        report[label] = design_sim
    return report


def find_mtd(toxicity_target, scenario, strictly_lte=False, verbose=False):
    """Finds the MTD in a list of toxicity probabilities.

    Args:
        toxicity_target (float): The target probability of toxicity.
        scenario (list[float]): A list of toxicity probabilities for each dose.
        strictly_lte (bool, optional): If `True`, the MTD must have a
            toxicity probability less than or equal to the target.
            Defaults to `False`.
        verbose (bool, optional): If `True`, prints extra information.
            Defaults to `False`.

    Returns:
        int: The 1-based index of the MTD.
    """
    if toxicity_target in scenario:
        # Return exact match
        loc = scenario.index(toxicity_target) + 1
        if verbose:
            logger.info("MTD is %s", loc)
        return loc
    else:
        if strictly_lte:
            if sum(np.array(scenario) <= toxicity_target) == 0:
                # Infeasible scenario
                if verbose:
                    logger.warning("All doses are too toxic")
                return 0
            else:
                # Return highest tox no greater than target
                objective = np.where(
                    np.array(scenario) <= toxicity_target,
                    toxicity_target - np.array(scenario),
                    np.inf,
                )
                loc = np.argmin(objective) + 1
                if verbose:
                    logger.info("Highest dose below MTD is %s", loc)
                return loc
        else:
            # Return nearest
            loc = np.argmin(np.abs(np.array(scenario) - toxicity_target)) + 1
            if verbose:
                logger.info("Dose nearest to MTD is %s", loc)
            return loc


def summarise_dose_finding_sims(sims, label, num_doses, filter={}):
    """Summarises a list of dose-finding simulations.

    .. deprecated:: 0.1.4
       Use `clintrials.simulation` functions instead.

    Args:
        sims (list[dict]): A list of simulation results.
        label (str): The label for the simulation set.
        num_doses (int): The number of dose levels.
        filter (dict, optional): A dictionary to filter the simulations.
            Defaults to {}.

    Returns:
        tuple: A tuple containing a DataFrame of dose information, a
            DataFrame of outcome information, an array of chosen doses, an
            array of doses given to patients, and an array of trial end
            statuses.
    """
    import pandas as pd

    if len(filter):
        sims = filter_list_of_dicts(sims, filter)

    # Recommended Doses
    doses = [x[label]["RecommendedDose"] for x in sims]
    df_doses = pd.DataFrame(
        {"RecN": pd.Series(doses).value_counts()}, index=range(-1, num_doses + 1)
    )
    df_doses.RecN[np.isnan(df_doses.RecN)] = 0
    df_doses["Rec%"] = 1.0 * df_doses["RecN"] / df_doses["RecN"].sum()
    # Given Doses
    doses_given = to_1d_list([x[label]["Doses"] for x in sims])
    df_doses = df_doses.join(
        pd.DataFrame({"PatN": pd.Series(doses_given).value_counts()})
    )
    df_doses.PatN[np.isnan(df_doses.PatN)] = 0
    df_doses["Pat%"] = 1.0 * df_doses["PatN"] / df_doses["PatN"].sum()
    df_doses["MeanPat"] = 1.0 * df_doses["PatN"] / len(sims)
    # Order
    df_doses = df_doses.loc[range(-1, num_doses + 1)]

    # Trial Outcomes
    statuses = [x[label]["TrialStatus"] for x in sims]
    df_statuses = pd.DataFrame({"N": pd.Series(statuses).value_counts()})
    df_statuses["%"] = 1.0 * df_statuses["N"] / df_statuses["N"].sum()

    return (
        df_doses,
        df_statuses,
        np.array(doses),
        np.array(doses_given),
        np.array(statuses),
    )


def batch_summarise_dose_finding_sims(
    sims, label, num_doses, dimensions=None, func1=None
):
    """Batch summarises a list of dose-finding simulations.

    .. deprecated:: 0.1.4
       Use `clintrials.simulation` functions instead.

    Args:
        sims (list[dict]): A list of simulation results.
        label (str): The label for the simulation set.
        num_doses (int): The number of dose levels.
        dimensions (tuple, optional): A tuple containing a dictionary mapping
            JSON variable names to parameter names and a `ParameterSpace`
            object. Defaults to `None`.
        func1 (callable, optional): A function to apply to the summary
            DataFrame. Defaults to `None`.
    """
    if dimensions is not None:
        var_map, params = dimensions
        z = [(k, params[v]) for k, v in var_map.items()]
        labels, val_arrays = zip(*z)
        param_combinations = list(product(*val_arrays))
        for param_combo in param_combinations:
            for lab, vals in zip(labels, param_combo):
                logger.info("%s: %s", lab, vals)
            these_params = dict(zip(labels, param_combo))
            abc = summarise_dose_finding_sims(
                sims, label, num_doses, filter=these_params
            )
            if func1:
                logger.info(func1(abc[0], these_params))
                logger.info("")
                logger.info("")
            else:
                logger.info("")
                logger.info(abc[0])
                logger.info("")
                logger.info(abc[1])
                logger.info("")
    else:
        abc = summarise_dose_finding_sims(sims, label, num_doses)
        logger.info(abc[0])
        logger.info("")
        logger.info(abc[1])
        logger.info("")


def dose_transition_pathways_to_json(
    trial,
    next_dose,
    cohort_sizes,
    cohort_number=1,
    cases_already_observed=[],
    custom_output_func=None,
    verbose=False,
    **kwargs,
):
    """Calculates the dose-transition pathways of a dose-finding trial.

    Args:
        trial (DoseFindingTrial): The trial design to use.
        next_dose (int): The dose for the next cohort.
        cohort_sizes (list[int]): A list of future cohort sizes.
        cohort_number (int, optional): The starting cohort number.
            Defaults to 1.
        cases_already_observed (list[tuple], optional): A list of previously
            observed cases. Defaults to [].
        custom_output_func (callable, optional): A function that takes the
            trial object and returns a dictionary of extra output.
            Defaults to `None`.
        verbose (bool, optional): If `True`, prints progress information.
            Defaults to `False`.
        **kwargs: Additional keyword arguments for the `trial.update`
            method.

    Returns:
        dict: A nested dictionary representing the dose-transition pathways.
    """
    if len(cohort_sizes) <= 0:
        return None
    else:
        cohort_size = cohort_sizes[0]

        path_outputs = []
        possible_dlts = range(0, cohort_size + 1)

        for i, num_dlts in enumerate(possible_dlts):

            # Invoke dose-decision
            cohort_cases = [(next_dose, 1)] * num_dlts + [(next_dose, 0)] * (
                cohort_size - num_dlts
            )
            cases = cases_already_observed + cohort_cases
            if verbose:
                logger.debug("Running %s", cases)
            trial.reset()
            # print 'next_dose is', trial.next_dose()
            trial.set_next_dose(next_dose)
            # print 'Now next_dose is', trial.next_dose()
            mtd = trial.update(cases, **kwargs)
            # print 'And now next_dose is', trial.next_dose()

            # Or:
            # mtd = trial.update(cases_already_observed, **kwargs)
            # trial.set_next_dose(next_dose)
            # mtd = trial.update(cohort_cases, **kwargs)

            # Collect output
            bag_o_tricks = OrderedDict(
                [
                    (f"Pat{cohort_number}.{j+1}", "Tox" if tox else "No Tox")
                    for (j, (dose, tox)) in enumerate(cohort_cases)
                ]
            )

            bag_o_tricks.update(
                OrderedDict(
                    [
                        ("DoseGiven", atomic_to_json(next_dose)),
                        ("RecommendedDose", atomic_to_json(mtd)),
                        ("CohortSize", cohort_size),
                        ("NumTox", atomic_to_json(num_dlts)),
                    ]
                )
            )
            if custom_output_func:
                bag_o_tricks.update(custom_output_func(trial))

            # Recurse subsequent cohorts
            further_paths = dose_transition_pathways_to_json(
                trial,
                next_dose=mtd,
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


dose_transition_pathways = dose_transition_pathways_to_json


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
        num_tox = x["NumTox"]
        mtd = x["RecommendedDose"]

        template_txt = "\t" * indent + "{} -> Dose {}"
        logger.info(template_txt.format(num_tox, dose_label_func(mtd)))

        if "Next" in x:
            print_dtps(x["Next"], indent=indent + 1, dose_label_func=dose_label_func)


def _dtps_to_rows(dtps, dose_label_func=None, pre=[]):
    """Converts DTPs to a list of rows for a DataFrame."""
    if dose_label_func is None:
        dose_label_func = lambda x: x
    rows = []
    for x in dtps:
        this_row = copy.copy(pre)
        num_tox = x["NumTox"]
        mtd = dose_label_func(x["RecommendedDose"])
        this_row.extend([num_tox, mtd])

        if "Next" in x:
            news_rows = _dtps_to_rows(
                x["Next"], dose_label_func=dose_label_func, pre=this_row
            )
            rows.extend(news_rows)
        else:
            rows.append(this_row)
    return rows


def dtps_to_pandas(dtps, dose_label_func=None):
    """Converts DTPs to a pandas DataFrame.

    Args:
        dtps (dict): A nested dictionary of DTPs.
        dose_label_func (callable, optional): A function to format the dose
            label. Defaults to `str`.

    Returns:
        pandas.DataFrame: A DataFrame representing the DTPs.
    """
    import pandas as pd

    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    rows = _dtps_to_rows(dtps, dose_label_func=dose_label_func)
    df = pd.DataFrame(rows)
    ncols = df.shape[1]
    cols = []
    for i in range(1, 1 + int(ncols / 2)):
        cols.extend([f"Cohort {i} DLTs", f"Cohort {i+1} Dose"])
    df.columns = cols

    return df
