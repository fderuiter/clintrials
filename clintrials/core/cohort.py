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
        """Adds a list of patients while ensuring the lists match in length."""
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
        """Adds standard patient records directly to the tracker."""
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
    """Parses various formats of cases into standardized PatientRecord objects."""
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


