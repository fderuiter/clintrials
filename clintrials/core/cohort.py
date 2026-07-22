"""Module containing the shared cohort tracking utility for trials."""

from dataclasses import dataclass
from typing import Iterable, List, Optional

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

