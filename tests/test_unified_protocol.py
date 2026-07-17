import pytest

from clintrials.core.protocol import Protocol
from clintrials.core.unified import SimulationResult


class DummyTrial(Protocol):
    def __init__(self):
        super().__init__()
        self.data = []
        self._step = 0

    def reset(self):
        self.data = []
        self._step = 0

    def update(self):
        # Draw a random number using injected RNG
        val = self.rng.normal(0, 1)
        self.data.append(val)
        self._step += 1

    def has_more(self):
        return self._step < 5

    def report(self):
        return {"sum": sum(self.data)}

    def run_bulk(self, n_sims: int, **kwargs):
        # Simulate vectorized behavior
        data = self.rng.normal(0, 1, size=(n_sims, 5))
        sums = data.sum(axis=1)
        return [{"sum": s} for s in sums]


def test_unified_protocol():
    # Test Iterative
    trial_iter = DummyTrial()
    res_iter = trial_iter.run(n_sims=100, method="iterative", seed=42)

    # Test Bulk
    trial_bulk = DummyTrial()
    res_bulk = trial_bulk.run(n_sims=100, method="bulk", seed=42)

    # Assert result container and identity
    assert isinstance(res_iter, SimulationResult)
    assert isinstance(res_bulk, SimulationResult)
    assert res_iter.mode == "iterative"
    assert res_bulk.mode == "bulk"

def test_simulation_result_dict_methods():
    res = SimulationResult({"a": 1, "b": 2}, mode="bulk")
    assert res.get("a") == 1
    assert res.get("c", 3) == 3
    assert list(res.keys()) == ["a", "b"]
    assert list(res.values()) == [1, 2]
    assert list(res.items()) == [("a", 1), ("b", 2)]
    assert res.to_list() == [{"a": 1, "b": 2}]

def test_simulation_result_list_methods():
    res = SimulationResult([1, 2, 3], mode="iterative")
    assert len(res) == 3
    assert res[1] == 2
    assert list(iter(res)) == [1, 2, 3]
    assert res.to_list() == [1, 2, 3]

    with pytest.raises(AttributeError):
        res.get("a")
    with pytest.raises(AttributeError):
        res.keys()
    with pytest.raises(AttributeError):
        res.values()
    with pytest.raises(AttributeError):
        res.items()
