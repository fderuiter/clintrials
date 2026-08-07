# SPDX-License-Identifier: MIT

"""Dose finding packages and core escalation trial structures."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"


__all__ = [
    "crm",
    "efftox",
    "efficacytoxicity",
    "wagestait",
    "watu",
    "DoseFindingTrial",
    "SimpleToxicityCountingDoseEscalationTrial",
    "ThreePlusThree",
    "simulate_dose_finding_trial",
    "simulate_dose_finding_trials",
    "find_mtd",
    "dose_transition_pathways_to_json",
    "dose_transition_pathways",
    "print_dtps",
    "dtps_to_pandas",
]


import abc
import copy
import logging
from collections import OrderedDict

from scipy.stats import uniform as uniform

from clintrials._utils import (
    atomic_to_json,
    iterable_to_json,
)

logger = logging.getLogger(__name__)


from clintrials.core.protocol import BaseDoseFindingTrial


class DoseFindingTrial(BaseDoseFindingTrial):
    """Base class for a dose-finding trial.

    Warning:
        Data updates are strictly incremental. Do not repeatedly pass the full
        patient history to the `update` method, as this will append duplicates
        to the internal records rather than replacing them. If you need to
        reload the full history (e.g., for data corrections), use the `reset()`
        method first.
    """

    def __init__(self, *, first_dose: int, num_doses: int, max_size: int) -> None:
        """Initializes a DoseFindingTrial object.

        Args:
            first_dose (int): The starting dose level (1-based).
            num_doses (int): The total number of dose levels.
            max_size (int): The maximum number of patients in the trial.

        Raises:
            ValueError: If `first_dose` is greater than `num_doses`.
        """
        super().__init__(first_dose=first_dose, num_doses=num_doses, max_size=max_size)

    def tabulate(self) -> pd.DataFrame:
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

    def update(self, cases: List[Any], **kwargs: Any) -> int:  # type: ignore[override]
        """Updates the trial with a list of new cases.

        Warning:
            This method is strictly incremental. It appends the provided cases
            to the internal record rather than replacing existing records.
            Calling `update()` with the full patient history repeatedly will
            duplicate existing records.

        Args:
            cases (list): A list of new cases to append.
            **kwargs (Any): Additional keyword arguments passed to the update logic.

        Returns:
            int: The next recommended dose level.
        """
        if cases:
            from clintrials.core.cohort import parse_patient_records

            records = parse_patient_records(cases)
            self._tracker.add_records(records)

        self._next_dose = self._calculate_next_dose(**kwargs)
        return self._next_dose

    def optimal_decision(self, prob_tox: Sequence[float]) -> int:
        """Gets the optimal dose choice for a given dose-toxicity curve.

        Args:
            prob_tox (list[float]): A list of toxicity probabilities for each
                dose level.

        Returns:
            int: The optimal 1-based dose level.
        """
        raise NotImplementedError()

    def plot_outcomes(self, chart_title: Optional[str] = None) -> Any:
        """Plots the outcomes of patients observed.

        Args:
            chart_title (str, optional): The title for the chart. Defaults to
                a descriptive title.

        Returns:
            A plot object.
        """
        from clintrials.core.viz_interface import get_visualization_provider

        viz = get_visualization_provider()  # type: ignore

        return viz.plot_dose_finding_outcomes(self, chart_title=chart_title)

    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return (self.size() < self.max_size()) and (self._status >= 0)

    def report(self) -> Dict[str, Any]:
        """Generates a standardized JSON-serializable report of the trial.

        Returns:
            collections.OrderedDict: The trial outcome report.
        """
        from collections import OrderedDict

        from clintrials._utils import atomic_to_json, iterable_to_json

        report = OrderedDict()
        report["RecommendedDose"] = atomic_to_json(self.next_dose())
        report["TrialStatus"] = atomic_to_json(self.status())
        report["Doses"] = iterable_to_json(self.doses())
        report["Toxicities"] = iterable_to_json(self.toxicities())
        return report

    @abc.abstractmethod
    def _calculate_next_dose(self, **kwargs: Any) -> int:
        """Calculates the next dose to be administered."""
        return -1  # Default implementation


class SimpleToxicityCountingDoseEscalationTrial(DoseFindingTrial):
    """A simple design that monotonically increases the dose.

    Escalates until a certain number of toxicities are observed.
    """

    def __init__(
        self, *, first_dose: Any, num_doses: Any, max_size: Any, max_toxicities: Any = 1
    ) -> None:
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

    def _reset(self) -> Any:
        self.max_dose_given = -1

    def _calculate_next_dose(self, **kwargs: Any) -> Any:
        if self.has_more():
            self._status = 1
            if len(self.doses()) > 0:
                return min(max(self.doses()) + 1, self.number_of_doses())
            else:
                return self._first_dose
        else:
            self._status = 100
            return max(self.doses())

    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        The trial stops if the maximum number of patients is reached, the
        maximum number of toxicities is observed, or the highest dose level
        is reached.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return (
            DoseFindingTrial.has_more(self)
            and (np.sum(self.toxicities()) < self.max_toxicities)
            and self.maximum_dose_given() < self.number_of_doses()  # type: ignore
        )


class ThreePlusThree(DoseFindingTrial):
    """An object-oriented implementation of the 3+3 trial design."""

    def __init__(self, *, num_doses: Any) -> None:
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

    def _reset(self) -> Any:
        self._continue = True

    def _calculate_next_dose(self, **kwargs: Any) -> Any:
        dose_indices = np.array(self._doses) == self._next_dose
        toxes_at_dose = np.sum(np.array(self._toxicities)[dose_indices])
        if np.sum(dose_indices) == 3:
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
        elif np.sum(dose_indices) == 6:
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

    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        The 3+3 trial stops when the MTD has been found.

        Returns:
            bool: `True` if the trial is ongoing, `False` otherwise.
        """
        return DoseFindingTrial.has_more(self) and self._continue


def _df_outcome_generator(
    design: Any,
    current_size: Any,
    cohort_size: Any,
    true_toxicities: Any,
    tolerances: Any,
    **kwargs: Any,
) -> Any:
    dose_level = design.next_dose()
    tox = [
        1 if x < true_toxicities[dose_level - 1] else 0
        for x in tolerances[current_size : current_size + cohort_size]
    ]
    return list(zip([dose_level] * cohort_size, tox))


def simulate_dose_finding_trial(
    design: Any,
    true_toxicities: Any,
    true_efficacies: Optional[Any] = None,
    tox_eff_odds_ratio: Any = 1.0,
    tolerances: Any = None,
    cohort_size: Any = 1,
    conduct_trial: Any = True,
    calculate_optimal_decision: Any = True,
    recruitment_stream: Any = None,
) -> Any:
    """Simulates a dose-finding trial.

    Args:
        design (DoseFindingTrial): The trial design to use.
        true_toxicities (list[float]): The true toxicity rates for each
            dose level.
        true_efficacies (list[float], optional): The true efficacy rates for each
            dose level (for joint models).
        tox_eff_odds_ratio (float, optional): Odds ratio for joint models.
        tolerances (list[float], optional): A list of uniform random
            numbers for simulating patient outcomes. Defaults to `None`.
        cohort_size (int, optional): The number of patients per cohort.
            Defaults to 1.
        conduct_trial (bool, optional): If `True`, conducts the trial
            cohort-by-cohort. Defaults to `True`.
        calculate_optimal_decision (bool, optional): If `True`, calculates
            the optimal dose decision. Defaults to `True`.
        recruitment_stream (RecruitmentStream, optional): A recruitment
            stream for patient arrival modeling. Defaults to None.

    Returns:
        collections.OrderedDict: A dictionary containing the simulation report.
    """
    from clintrials.core.simulation import UniversalProtocolSimulationRunner
    from clintrials.dosefinding.efficacytoxicity import (
        EfficacyToxicityDoseFindingTrial,
        _simulate_trial,
    )

    if (
        isinstance(design, EfficacyToxicityDoseFindingTrial)
        or true_efficacies is not None
    ):
        if true_efficacies is None:
            raise ValueError(
                "true_efficacies must be provided for joint efficacy-toxicity designs."
            )
        n_patients = design.max_size()
        if tolerances is not None:
            if isinstance(tolerances, np.ndarray) and tolerances.ndim == 2:
                pass
            else:
                flat_tols = list(tolerances)
                if len(flat_tols) >= 3 * n_patients:
                    tolerances = np.array(flat_tols[: 3 * n_patients]).reshape(
                        n_patients, 3
                    )
                else:
                    tolerances = np.random.uniform(size=3 * n_patients).reshape(
                        n_patients, 3
                    )
        else:
            tolerances = np.random.uniform(size=3 * n_patients).reshape(n_patients, 3)

        return _simulate_trial(
            design=design,
            true_toxicities=true_toxicities,
            true_efficacies=true_efficacies,
            tox_eff_odds_ratio=tox_eff_odds_ratio,
            tolerances=tolerances,
            cohort_size=cohort_size,
            conduct_trial=conduct_trial,
            calculate_optimal_decision=calculate_optimal_decision,
            recruitment_stream=recruitment_stream,
        )

    # Validate inputs for single-endpoint trials
    if tolerances is None:
        tolerances = uniform().rvs(design.max_size())
    else:
        if isinstance(tolerances, np.ndarray) and tolerances.ndim > 1:
            tolerances = tolerances.flatten()
        if len(tolerances) < design.max_size():
            logging.warning(
                "You have provided fewer tolerances than maximum number of patients on trial. Beware errors!"
            )

    report = OrderedDict()
    report["TrueToxicities"] = iterable_to_json(true_toxicities)

    # Simulate trial
    if conduct_trial:
        runner = UniversalProtocolSimulationRunner(
            design=design,
            outcome_generator=_df_outcome_generator,  # type: ignore[arg-type]
            recruitment_stream=recruitment_stream,
        )
        sim_report = runner.run(
            cohort_size=cohort_size,
            true_toxicities=true_toxicities,
            tolerances=tolerances,
        )
        if isinstance(sim_report, list):
            sim_report = sim_report[0]
        report.update(sim_report)
    else:
        report.update(design.report())

    # Optimal decision, given these specific patient tolerances
    if calculate_optimal_decision:
        try:
            had_tox = lambda x: x < np.array(true_toxicities)
            tox_horizons = np.array(
                [had_tox(x) for x in tolerances[: design.max_size()]]
            )  # type: ignore
            tox_hat = tox_horizons.mean(axis=0)

            optimal_allocation = design.optimal_decision(tox_hat)
            report["FullyInformedToxicityCurve"] = iterable_to_json(tox_hat)
            report["OptimalAllocation"] = atomic_to_json(optimal_allocation)
        except NotImplementedError:
            pass

    return report


def simulate_dose_finding_trials(
    design_map: Any,
    true_toxicities: Any,
    true_efficacies: Optional[Any] = None,
    tox_eff_odds_ratio: Any = 1.0,
    tolerances: Any = None,
    cohort_size: Any = 1,
    conduct_trial: Any = True,
    calculate_optimal_decision: Any = True,
    recruitment_stream: Any = None,
) -> Any:
    """Simulates multiple toxicity-driven dose-finding trials.

    Runs simulations from the same patient data.

    Args:
        design_map (dict[str, DoseFindingTrial]): A dictionary mapping
            design labels to trial design objects.
        true_toxicities (list[float]): The true toxicity rates for each
            dose level.
        true_efficacies (list[float], optional): True efficacy rates.
        tox_eff_odds_ratio (float, optional): Odds ratio for joint models.
        tolerances (list[float], optional): A list of uniform random
            numbers for simulating patient outcomes. Defaults to `None`.
        cohort_size (int, optional): The number of patients per cohort.
            Defaults to 1.
        conduct_trial (bool, optional): If `True`, conducts the trial
            cohort-by-cohort. Defaults to `True`.
        calculate_optimal_decision (bool, optional): If `True`, calculates
            the optimal dose decision. Defaults to `True`.
        recruitment_stream (RecruitmentStream, optional): A recruitment
            stream for patient arrival modeling. Defaults to None.

    Returns:
        collections.OrderedDict: A dictionary of simulation reports, with
            keys corresponding to the design labels.
    """
    from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial

    has_joint = any(
        isinstance(design, EfficacyToxicityDoseFindingTrial)
        for design in design_map.values()
    )
    max_size = max([design.max_size() for design in design_map.values()])

    if tolerances is None:
        if has_joint:
            tolerances = np.random.uniform(size=3 * max_size).reshape(max_size, 3)
        else:
            tolerances = uniform().rvs(max_size)

    report = OrderedDict()
    report["TrueToxicities"] = iterable_to_json(true_toxicities)
    if has_joint and true_efficacies is not None:
        report["TrueEfficacies"] = iterable_to_json(true_efficacies)

    for label, design in design_map.items():
        design_sim = simulate_dose_finding_trial(
            design=design,
            true_toxicities=true_toxicities,
            true_efficacies=true_efficacies,
            tox_eff_odds_ratio=tox_eff_odds_ratio,
            tolerances=tolerances,
            cohort_size=cohort_size,
            conduct_trial=conduct_trial,
            calculate_optimal_decision=calculate_optimal_decision,
            recruitment_stream=recruitment_stream,
        )
        report[label] = design_sim
    return report


def find_mtd(
    toxicity_target: Any, scenario: Any, strictly_lte: Any = False, verbose: Any = False
) -> Any:
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
            if np.sum(np.array(scenario) <= toxicity_target) == 0:
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


def dose_transition_pathways_to_json(
    trial: DoseFindingTrial,
    next_dose: int,
    cohort_sizes: List[int],
    cohort_number: int = 1,
    cases_already_observed: List[Tuple] = [],
    custom_output_func: Optional[Callable] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:  # type: ignore
    """Calculates the dose-transition pathways of a dose-finding trial.

    Args:
        trial (DoseFindingTrial): The trial design to use.
        next_dose (int): The dose for the next cohort.
        cohort_sizes (list[int]): A list of future cohort sizes.
        cohort_number (int, optional): The starting cohort number.
            Defaults to 1.
        cases_already_observed (list[tuple], optional): A list of previously
            observed cases. Defaults to [].
        custom_output_func (Callable, optional): A function that takes the
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
            from clintrials.core.cohort import parse_patient_records

            mtd = trial.update(parse_patient_records(cases), **kwargs)
            # print 'And now next_dose is', trial.next_dose()

            # Or:
            # mtd = trial.update(cases_already_observed, **kwargs)
            # trial.set_next_dose(next_dose)
            # mtd = trial.update(cohort_cases, **kwargs)

            # Collect output
            bag_o_tricks = OrderedDict(
                [
                    (f"Pat{cohort_number}.{j + 1}", "Tox" if tox else "No Tox")
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


def print_dtps(
    dtps: Any,
    indent: int = 0,
    dose_label_func: Optional[Callable] = None,
    row_formatter: Optional[Callable] = None,
    verbose: bool = False,
) -> Any:  # type: ignore
    """Prints the dose-transition pathways.

    Args:
        dtps (dict): A nested dictionary of DTPs.
        indent (int, optional): The indentation level. Defaults to 0.
        dose_label_func (Callable, optional): A function to format the dose
            label. Defaults to `str`.
        row_formatter (Callable, optional): A function to format the row.
        verbose (bool, optional): Whether to print verbose output.
    """
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    for x in dtps:
        if row_formatter:
            row_str = row_formatter(x, dose_label_func=dose_label_func, verbose=verbose)
        else:
            num_tox = x["NumTox"]
            mtd = x["RecommendedDose"]
            row_str = f"{num_tox} -> Dose {dose_label_func(mtd)}"

        prefix = "  " * indent + "- "
        logger.info(prefix + row_str)

        if "Next" in x:
            print_dtps(
                x["Next"],
                indent=indent + 1,
                dose_label_func=dose_label_func,
                row_formatter=row_formatter,
                verbose=verbose,
            )


def _dtps_to_rows(dtps: Any, dose_label_func: Any = None, pre: Any = []) -> Any:
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


def dtps_to_pandas(dtps: Any, dose_label_func: Optional[Callable] = None) -> Any:  # type: ignore
    """Converts DTPs to a pandas DataFrame.

    Args:
        dtps (dict): A nested dictionary of DTPs.
        dose_label_func (Callable, optional): A function to format the dose
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
        cols.extend([f"Cohort {i} DLTs", f"Cohort {i + 1} Dose"])
    df.columns = cols

    return df
