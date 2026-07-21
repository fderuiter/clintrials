import pytest

from clintrials.dosefinding import ThreePlusThree


def test_simple_escalation():  # type: ignore
    t = ThreePlusThree(3)  # type: ignore
    assert t.next_dose() == 1
    # first cohort at dose 1, no tox -> escalate
    next_dose = t.update([(1, 0), (1, 0), (1, 0)])
    assert next_dose == 2
    assert t.has_more()


def test_hold_after_one_toxicity():  # type: ignore
    t = ThreePlusThree(3)  # type: ignore
    t.update([(1, 0), (1, 0), (1, 0)])  # escalate to 2
    # one toxicity in cohort -> stay on same dose
    next_dose = t.update([(2, 1), (2, 0), (2, 0)])
    assert next_dose == 2
    assert t.has_more()


def test_deescalate_and_stop():  # type: ignore
    t = ThreePlusThree(2)  # type: ignore
    t.update([(1, 0), (1, 0), (1, 0)])  # escalate to 2
    # two toxicities triggers de-escalation and stop
    next_dose = t.update([(2, 1), (2, 1), (2, 0)])
    assert next_dose == 1
    assert not t.has_more()


def test_invalid_batch_size_raises():  # type: ignore
    t = ThreePlusThree(2)  # type: ignore
    with pytest.raises(Exception):
        t.update([(1, 0)])
