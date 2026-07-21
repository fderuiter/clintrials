import numpy as np
import pytest

from clintrials.core.math import (
    empiric,
    inverse_empiric,
    inverse_logistic,
    inverse_logit1,
    logistic,
    logit1,
)


def test_empiric_accuracy():  # type: ignore
    # empiric(x, beta=0) -> x^1 = x
    assert empiric(0.5, beta=0) == pytest.approx(0.5)
    # empiric(0.5, beta=np.log(2)) -> 0.5^2 = 0.25
    assert empiric(0.5, beta=np.log(2)) == pytest.approx(0.25)


def test_inverse_empiric_accuracy():  # type: ignore
    # inverse_empiric(x, beta=0) -> x^1 = x
    assert inverse_empiric(0.25, beta=0) == pytest.approx(0.25)
    # inverse_empiric(0.25, beta=np.log(2)) -> 0.25^0.5 = 0.5
    assert inverse_empiric(0.25, beta=np.log(2)) == pytest.approx(0.5)


def test_logistic_accuracy():  # type: ignore
    # logistic(x, a0=0, beta=0) -> 1 / (1 + exp(-x))
    expected = 1 / (1 + np.exp(-0.25))
    assert logistic(0.25, a0=0, beta=0) == pytest.approx(expected)


def test_inverse_logistic_accuracy():  # type: ignore
    y = logistic(0.25, a0=-1, beta=1)
    x_inv = inverse_logistic(y, a0=-1, beta=1)
    assert x_inv == pytest.approx(0.25)


def test_logit1_accuracy():  # type: ignore
    # logit1 uses a0=3 default
    expected = 1 / (1 + np.exp(-3 - np.exp(0) * 0.25))
    assert logit1(0.25) == pytest.approx(expected)  # type: ignore
    assert logit1(0.25, beta=0) == pytest.approx(expected)  # type: ignore


def test_inverse_logit1_accuracy():  # type: ignore
    y = logit1(0.25)  # type: ignore
    x_inv = inverse_logit1(y)  # type: ignore
    assert x_inv == pytest.approx(0.25)
