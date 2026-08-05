import pytest
import numpy as np
from clintrials.core.cohort import PatientRecord, PatientCohortTracker, parse_patient_records


def test_patient_record_initialization() -> None:
    """Test PatientRecord class initialization and defaults."""
    record = PatientRecord(dose=2, toxicity=1)
    assert record.dose == 2
    assert record.toxicity == 1
    assert record.efficacy is None

    record_with_eff = PatientRecord(dose=3, toxicity=0, efficacy=1)
    assert record_with_eff.dose == 3
    assert record_with_eff.toxicity == 0
    assert record_with_eff.efficacy == 1


def test_patient_cohort_tracker_add_patients() -> None:
    """Test PatientCohortTracker adding patients with matching lists."""
    tracker = PatientCohortTracker()
    assert len(tracker) == 0

    # Add single-endpoint patients
    tracker.add_patients(doses=[1, 2, 2], toxicities=[0, 1, 0])
    assert len(tracker) == 3
    assert tracker.doses == [1, 2, 2]
    assert tracker.toxicities == [0, 1, 0]
    assert tracker.efficacies == []

    # Reset tracker
    tracker.reset()
    assert len(tracker) == 0

    # Add dual-endpoint patients
    tracker.add_patients(doses=[1, 3], toxicities=[0, 0], efficacies=[1, 0])
    assert len(tracker) == 2
    assert tracker.doses == [1, 3]
    assert tracker.toxicities == [0, 0]
    assert tracker.efficacies == [1, 0]


def test_patient_cohort_tracker_add_patients_mismatch_raises() -> None:
    """Test PatientCohortTracker raises ValueError when lengths mismatch."""
    tracker = PatientCohortTracker()
    with pytest.raises(ValueError):
        tracker.add_patients(doses=[1, 2], toxicities=[0])

    with pytest.raises(ValueError):
        tracker.add_patients(doses=[1, 2], toxicities=[0, 0], efficacies=[1])


def test_patient_cohort_tracker_add_records() -> None:
    """Test PatientCohortTracker adding standard records directly."""
    tracker = PatientCohortTracker()
    records = [
        PatientRecord(dose=1, toxicity=0),
        PatientRecord(dose=2, toxicity=1, efficacy=1),
    ]
    tracker.add_records(records)
    assert len(tracker) == 2
    assert tracker.doses == [1, 2]
    assert tracker.toxicities == [0, 1]
    assert tracker.efficacies == [1]


def test_parse_patient_records_various_formats() -> None:
    """Test parsing of different input types/formats with parse_patient_records."""
    # None and empty
    assert parse_patient_records(None) == []
    assert parse_patient_records([]) == []

    # Single PatientRecord
    rec = PatientRecord(dose=1, toxicity=0)
    assert parse_patient_records(rec) == [rec]

    # Single dictionary
    assert parse_patient_records({"dose": 2, "toxicity": 1}) == [PatientRecord(dose=2, toxicity=1)]
    assert parse_patient_records({"dose": 2, "toxicity": 1, "efficacy": 0}) == [
        PatientRecord(dose=2, toxicity=1, efficacy=0)
    ]

    # Single 2-tuple and 3-tuple
    assert parse_patient_records((1, 0)) == [PatientRecord(dose=1, toxicity=0)]
    assert parse_patient_records((2, 1, 0)) == [PatientRecord(dose=2, toxicity=1, efficacy=0)]

    # Tuple of tuples and list of tuples
    assert parse_patient_records([(1, 0), (2, 1, 1)]) == [
        PatientRecord(dose=1, toxicity=0),
        PatientRecord(dose=2, toxicity=1, efficacy=1),
    ]

    # NumPy arrays
    np_arr = np.array([[1, 0], [2, 1]])
    assert parse_patient_records(np_arr) == [
        PatientRecord(dose=1, toxicity=0),
        PatientRecord(dose=2, toxicity=1),
    ]


def test_parse_patient_records_invalid_formats() -> None:
    """Test parse_patient_records invalid input raises appropriate errors."""
    with pytest.raises(TypeError):
        parse_patient_records("invalid_format")

    with pytest.raises(ValueError):
        parse_patient_records([(1,)])  # Tuple of length 1 is invalid
