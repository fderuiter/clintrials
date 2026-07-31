# type: ignore
import pytest

from clintrials.dosefinding.efficacytoxicity import (
    EfficacyToxicityDoseFindingTrial,
    _efftox_patient_outcome_to_label,
)


class DummyEfficacyToxicityTrial(EfficacyToxicityDoseFindingTrial):
    def _calculate_next_dose(self, **kwargs):
        return 1

    def _reset(self):
        pass

    def has_more(self):
        return True


def test_efftox_patient_outcome_to_label():
    """Test translating outcome pairs to string labels."""
    assert _efftox_patient_outcome_to_label((0, 0)) == "Neither"
    assert _efftox_patient_outcome_to_label((1, 0)) == "Toxicity"
    assert _efftox_patient_outcome_to_label((0, 1)) == "Efficacy"
    assert _efftox_patient_outcome_to_label((1, 1)) == "Both"
    assert _efftox_patient_outcome_to_label((2, 2)) == "Error"


def test_efficacy_toxicity_dose_finding_trial_init_validation():
    """Test validation checks during trial initialization."""
    with pytest.raises(ValueError, match="First dose must be no greater than number of doses"):
        DummyEfficacyToxicityTrial(first_dose=5, num_doses=3, max_size=20)
