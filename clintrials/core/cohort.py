# SPDX-License-Identifier: MIT

"""Module containing the shared cohort tracking utility for trials."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from clintrials.validation import validate_matching_lengths


@dataclass
class PatientRecord:
    """A structured representation of a patient's outcomes."""
    dose: int
    toxicity: int
    efficacy: Optional[int] = None

class PatientCohortTracker:
    """A shared utility to track patient outcomes in unified records."""

    def __init__(self) -> None:
        """Initializes an empty PatientCohortTracker."""
        self.records: List[PatientRecord] = []

    def add_patients(
        self,
        doses: Iterable[int],
        toxicities: Iterable[int],
        efficacies: Optional[Iterable[int]] = None
    ) -> None:
        """Adds a list of patients while ensuring the lists match in length.

        This method validates that the input lists (doses, toxicities, and optionally
        efficacies) are of equal length. If a length mismatch is detected, a ValueError
        is raised and no state changes are committed to the tracker.

        Args:
            doses (Iterable[int]): The sequence of administered doses to add.
            toxicities (Iterable[int]): The sequence of toxicity outcomes matching the doses.
            efficacies (Optional[Iterable[int]], optional): The sequence of efficacy outcomes
                matching the doses. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the length of toxicities or efficacies does not match
                the length of doses.
        """
        doses_list = list(doses)
        toxicities_list = list(toxicities)

        # Validates matching lengths before any state changes occur
        if efficacies is not None:
            efficacies_list = list(efficacies)
            validate_matching_lengths(
                doses=doses_list,
                toxicities=toxicities_list,
                efficacies=efficacies_list
            )
            new_records = [
                PatientRecord(dose=d, toxicity=t, efficacy=e)
                for d, t, e in zip(doses_list, toxicities_list, efficacies_list)
            ]
        else:
            validate_matching_lengths(
                doses=doses_list,
                toxicities=toxicities_list
            )
            new_records = [
                PatientRecord(dose=d, toxicity=t)
                for d, t in zip(doses_list, toxicities_list)
            ]

        # Commit changes atomically
        self.records.extend(new_records)

    def reset(self) -> None:
        """Clears all records from the tracker."""
        self.records.clear()

    def add_records(self, records: Iterable[PatientRecord]) -> None:
        """Adds standard patient records directly to the tracker.

        Args:
            records (Iterable[PatientRecord]): The structured PatientRecord objects to add.

        Returns:
            None
        """
        self.records.extend(records)

    @property
    def doses(self) -> List[int]:
        """Gets a derived list of doses for all patients."""
        return [r.dose for r in self.records]

    @property
    def toxicities(self) -> List[int]:
        """Gets a derived list of toxicity outcomes for all patients."""
        return [r.toxicity for r in self.records]

    @property
    def efficacies(self) -> List[int]:
        """Gets a derived list of efficacy outcomes for all patients."""
        return [r.efficacy for r in self.records if r.efficacy is not None]

    def __len__(self) -> int:
        """Returns the total number of patients tracked."""
        return len(self.records)


def parse_patient_records(cases: Any) -> List[PatientRecord]:
    """Parses various formats of cases into standardized PatientRecord objects.

    Args:
        cases (Any): The patient cases to parse. Can be a single case or an iterable
            of cases. Supports PatientRecord objects, dicts, or tuples.

    Returns:
        List[PatientRecord]: A list of parsed PatientRecord objects.

    Raises:
        TypeError: If an unsupported case format is encountered.
        ValueError: If a tuple case does not have the expected length.
    """
    import numpy as np
    if cases is None:
        return []
    try:
        if len(cases) == 0:
            return []
    except TypeError:
        pass

    records = []
    # If a single item is passed instead of a list, wrap it
    if isinstance(cases, (PatientRecord, dict)):
        cases = [cases]
    elif isinstance(cases, tuple):
        # Could be a single tuple like (1, 0) or (1, 0, 1), or a tuple of tuples.
        # Let's check if the first element is a number or not.
        if len(cases) > 0 and isinstance(cases[0], (int, float, np.integer, np.floating)):
            cases = [cases]

    for case in cases:
        if isinstance(case, PatientRecord):
            records.append(case)
        elif isinstance(case, dict):
            dose = int(case["dose"])
            toxicity = int(case["toxicity"])
            efficacy = case.get("efficacy")
            if efficacy is not None:
                efficacy = int(efficacy)
            records.append(PatientRecord(dose=dose, toxicity=toxicity, efficacy=efficacy))
        elif isinstance(case, (tuple, list, np.ndarray)):
            # Convert to list or standard python tuple
            case_list = list(case)
            if len(case_list) == 2:
                records.append(PatientRecord(dose=int(case_list[0]), toxicity=int(case_list[1])))
            elif len(case_list) >= 3:
                # Could be 3 or more, e.g. (dose, toxicity, efficacy) or (dose, toxicity, efficacy, arrival_time, ...)
                # Let's extract the first three
                efficacy_val = case_list[2]
                if efficacy_val is not None:
                    efficacy_val = int(efficacy_val)
                records.append(
                    PatientRecord(
                        dose=int(case_list[0]),
                        toxicity=int(case_list[1]),
                        efficacy=efficacy_val
                    )
                )
            else:
                from clintrials.core.errors import ErrorTemplates
                raise ValueError(ErrorTemplates.EXPECTED_LENGTH.format(name="Patient outcome", expected_length=2))
        else:
            raise TypeError(f"Unsupported patient record format: {type(case)}")
    return records


