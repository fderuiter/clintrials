import pytest
from clintrials.dosefinding import ThreePlusThree


def test_simple_escalation():
    t = ThreePlusThree(3)
    assert t.next_dose() == 1
    # first cohort at dose 1, no tox -> escalate
    next_dose = t.update([(1, 0), (1, 0), (1, 0)])
    assert next_dose == 2
    assert t.has_more()


def test_hold_after_one_toxicity():
    t = ThreePlusThree(3)
    t.update([(1, 0), (1, 0), (1, 0)])  # escalate to 2
    # one toxicity in cohort -> stay on same dose
    next_dose = t.update([(2, 1), (2, 0), (2, 0)])
    assert next_dose == 2
    assert t.has_more()


def test_deescalate_and_stop():
    t = ThreePlusThree(2)
    t.update([(1, 0), (1, 0), (1, 0)])  # escalate to 2
    # two toxicities triggers de-escalation and stop
    next_dose = t.update([(2, 1), (2, 1), (2, 0)])
    assert next_dose == 1
    assert not t.has_more()


def test_invalid_batch_size_raises():
    t = ThreePlusThree(2)
    with pytest.raises(Exception):
        t.update([(1, 0)])
