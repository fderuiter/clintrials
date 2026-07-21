import pytest

from clintrials.core.protocol import Protocol
from clintrials.core.unified import SimulationResult


class DummyTrial(Protocol):
    def __init__(self):  # type: ignore
        super().__init__()  # type: ignore
        self.data = []
        self._step = 0

    def reset(self):  # type: ignore
        self.data = []
        self._step = 0

    def update(self):  # type: ignore
        # Draw a random number using injected RNG
        val = self.rng.normal(0, 1)
        self.data.append(val)
        self._step += 1

    def has_more(self):  # type: ignore
        return self._step < 5

    def report(self):  # type: ignore
        return {"sum": sum(self.data)}

    def run_bulk(self, n_sims: int, **kwargs):  # type: ignore
        # Simulate vectorized behavior
        data = self.rng.normal(0, 1, size=(n_sims, 5))
        sums = data.sum(axis=1)
        return [{"sum": s} for s in sums]


def test_unified_protocol():  # type: ignore
    # Test Iterative
    trial_iter = DummyTrial()  # type: ignore
    res_iter = trial_iter.run(n_sims=100, method="iterative", seed=42)

    # Test Bulk
    trial_bulk = DummyTrial()  # type: ignore
    res_bulk = trial_bulk.run(n_sims=100, method="bulk", seed=42)

    # Assert result container and identity
    assert isinstance(res_iter, SimulationResult)
    assert isinstance(res_bulk, SimulationResult)
    assert res_iter.mode == "iterative"
    assert res_bulk.mode == "bulk"

def test_simulation_result_dict_methods():  # type: ignore
    res = SimulationResult({"a": 1, "b": 2}, mode="bulk")  # type: ignore
    assert res.get("a") == 1  # type: ignore
    assert res.get("c", 3) == 3  # type: ignore
    assert list(res.keys()) == ["a", "b"]  # type: ignore
    assert list(res.values()) == [1, 2]  # type: ignore
    assert list(res.items()) == [("a", 1), ("b", 2)]  # type: ignore
    assert res.to_list() == [{"a": 1, "b": 2}]  # type: ignore

def test_simulation_result_list_methods():  # type: ignore
    res = SimulationResult([1, 2, 3], mode="iterative")  # type: ignore
    assert len(res) == 3
    assert res[1] == 2
    assert list(iter(res)) == [1, 2, 3]
    assert res.to_list() == [1, 2, 3]  # type: ignore

    with pytest.raises(AttributeError):
        res.get("a")  # type: ignore
    with pytest.raises(AttributeError):
        res.keys()  # type: ignore
    with pytest.raises(AttributeError):
        res.values()  # type: ignore
    with pytest.raises(AttributeError):
        res.items()  # type: ignore
