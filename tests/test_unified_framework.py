# SPDX-License-Identifier: MIT

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
        return int(
            min(self._first_dose + len(self.doses()) * multiplier, self.num_doses)
        )


class CustomDualEndpointTrial(EfficacyToxicityDoseFindingTrial):
    """A custom dual-endpoint trial design implemented by overriding exactly two standard protected hooks."""

    def _reset(self) -> None:
        self.custom_reset_called = True
        self.custom_state = "initial"

    def _calculate_next_dose(self, **kwargs: Any) -> int:
        multiplier = int(kwargs.get("multiplier", 1))
        return int(
            min(self._first_dose + len(self.doses()) * multiplier, self.num_doses)
        )


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


def test_unified_patient_records_update_types() -> None:
    from clintrials.core.cohort import PatientRecord

    trial = CustomSingleEndpointTrial(first_dose=1, num_doses=5, max_size=10)

    # Test updating with PatientRecord objects directly
    records = [PatientRecord(dose=1, toxicity=0), PatientRecord(dose=1, toxicity=1)]
    trial.update(records)
    assert trial.doses() == [1, 1]
    assert trial.toxicities() == [0, 1]

    # Test updating with dicts
    dict_records = [
        {"dose": 2, "toxicity": 0},
        {"dose": 2, "toxicity": 1, "efficacy": 1},
    ]
    trial.update(dict_records)
    assert trial.doses() == [1, 1, 2, 2]
    assert trial.toxicities() == [0, 1, 0, 1]


def test_estimator_injection_and_forwarding() -> None:
    from clintrials.dosefinding.crm import CRM
    from clintrials.dosefinding.efftox import LpNormCurve
    from clintrials.dosefinding.watu import WATU

    # Instantiate custom CRM estimator
    custom_crm = CRM(
        prior=[0.05, 0.1, 0.2, 0.3, 0.4],
        target=0.25,
        first_dose=1,
        max_size=12,
    )

    # Inject it into WATU
    watu_trial = WATU(
        skeletons=[[0.1, 0.2, 0.3, 0.4, 0.5]],
        prior_tox_probs=[0.05, 0.1, 0.2, 0.3, 0.4],
        tox_target=0.25,
        tox_limit=0.3,
        eff_limit=0.2,
        metric=LpNormCurve(0.05, 0.4, 0.25, 0.15),
        first_dose=1,
        max_size=12,
        toxicity_estimator=custom_crm,
    )

    assert watu_trial.crm is custom_crm

    # Check that update succeeds without custom parameter-forwarding TypeErrors
    # even when extraneous parameters are forwarded.
    watu_trial.update([(1, 0, 1)], extraneous_param="ignored")
    assert watu_trial.crm.doses() == [1]


def test_consolidated_simulation_execution() -> None:
    from clintrials.dosefinding import simulate_dose_finding_trial
    from clintrials.dosefinding.crm import CRM
    from clintrials.dosefinding.efftox import LpNormCurve
    from clintrials.dosefinding.watu import WATU

    # CRM simulation
    crm_trial = CRM(
        prior=[0.05, 0.1, 0.2, 0.3, 0.4],
        target=0.25,
        first_dose=1,
        max_size=10,
    )
    crm_sim = simulate_dose_finding_trial(
        design=crm_trial,
        true_toxicities=[0.05, 0.1, 0.2, 0.3, 0.4],
        tolerances=[0.5] * 10,
        cohort_size=3,
    )
    assert "RecommendedDose" in crm_sim

    # WATU joint design simulation using the exact same pathway
    watu_trial = WATU(
        skeletons=[[0.1, 0.2, 0.3, 0.4, 0.5]],
        prior_tox_probs=[0.05, 0.1, 0.2, 0.3, 0.4],
        tox_target=0.25,
        tox_limit=0.3,
        eff_limit=0.2,
        metric=LpNormCurve(0.05, 0.4, 0.25, 0.15),
        first_dose=1,
        max_size=10,
    )
    watu_sim = simulate_dose_finding_trial(
        design=watu_trial,
        true_toxicities=[0.05, 0.1, 0.2, 0.3, 0.4],
        true_efficacies=[0.1, 0.2, 0.3, 0.4, 0.5],
        tolerances=[0.5] * 30,  # 3 * n_patients flat array
        cohort_size=3,
    )
    assert "RecommendedDose" in watu_sim
