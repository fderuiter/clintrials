from typing import Any

from clintrials.dosefinding import DoseFindingTrial
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial


class CustomSingleEndpointTrial(DoseFindingTrial):
    """A custom single-endpoint trial design implemented by overriding exactly two standard protected hooks."""

    def _reset(self) -> None:
        self.custom_reset_called = True
        self.custom_state = "initial"

    def _calculate_next_dose(self, **kwargs: Any) -> int:
        multiplier = int(kwargs.get("multiplier", 1))
        return int(min(self._first_dose + len(self.doses()) * multiplier, self.num_doses))


class CustomDualEndpointTrial(EfficacyToxicityDoseFindingTrial):
    """A custom dual-endpoint trial design implemented by overriding exactly two standard protected hooks."""

    def _reset(self) -> None:
        self.custom_reset_called = True
        self.custom_state = "initial"

    def _calculate_next_dose(self, **kwargs: Any) -> int:
        multiplier = int(kwargs.get("multiplier", 1))
        return int(min(self._first_dose + len(self.doses()) * multiplier, self.num_doses))


def test_custom_single_endpoint_trial_flow() -> None:
    # Initialize
    trial = CustomSingleEndpointTrial(first_dose=1, num_doses=5, max_size=10)

    # Check that custom reset is called during initialization or when reset is called
    trial.reset()
    assert trial.custom_reset_called is True
    assert trial.custom_state == "initial"

    # Check getters/properties
    assert trial.first_dose() == 1
    assert trial.number_of_doses() == 5
    assert trial.max_size() == 10
    assert trial.size() == 0
    assert list(trial.dose_levels()) == [1, 2, 3, 4, 5]

    # Update with some cases
    next_dose = trial.update([(1, 0), (1, 1)])
    assert next_dose == 3  # len(doses) is 2, first_dose + 2 = 3
    assert trial.size() == 2
    assert trial.doses() == [1, 1]
    assert trial.toxicities() == [0, 1]

    # Test update with optional kwargs parameter passing to _calculate_next_dose
    next_dose_with_param = trial.update([(3, 0)], multiplier=2)
    # len(doses) is now 3, first_dose is 1. Under multiplier=2, next recommended dose should be min(1 + 3*2, 5) = 5
    assert next_dose_with_param == 5
    assert trial.size() == 3

    # Verify report schema
    report = trial.report()
    assert report["RecommendedDose"] == 5
    assert report["TrialStatus"] == 0
    assert report["Doses"] == [1, 1, 3]
    assert report["Toxicities"] == [0, 1, 0]


def test_custom_dual_endpoint_trial_flow() -> None:
    # Initialize
    trial = CustomDualEndpointTrial(first_dose=1, num_doses=5, max_size=12)

    # Reset
    trial.reset()
    assert trial.custom_reset_called is True
    assert trial.custom_state == "initial"

    # Check getters/properties
    assert trial.first_dose() == 1
    assert trial.number_of_doses() == 5
    assert trial.max_size() == 12
    assert trial.size() == 0
    assert list(trial.dose_levels()) == [1, 2, 3, 4, 5]

    # Update with some cases (dose, toxicity, efficacy)
    next_dose = trial.update([(1, 0, 1), (1, 1, 0)])
    assert next_dose == 3  # len(doses) is 2, first_dose + 2 = 3
    assert trial.size() == 2
    assert trial.doses() == [1, 1]
    assert trial.toxicities() == [0, 1]
    assert trial.efficacies() == [1, 0]

    # Test update with optional kwargs parameter passing
    next_dose_with_param = trial.update([(3, 0, 1)], multiplier=2)
    assert next_dose_with_param == 5
    assert trial.size() == 3

    # Verify report schema
    report = trial.report()
    assert report["RecommendedDose"] == 5
    assert report["TrialStatus"] == 0
    assert report["Doses"] == [1, 1, 3]
    assert report["Toxicities"] == [0, 1, 0]
    assert report["Efficacies"] == [1, 0, 1]
