from clintrials.utils import Memoize


# Define a standard parameter class representing configuration data
class SimulationConfig:
    def __init__(self, n_trials: int, learning_rate: float):
        self.n_trials = n_trials
        self.learning_rate = learning_rate

# Define a custom dynamic callable object (implementing __call__)
class CustomCallable:
    def __init__(self, factor: int):
        self.factor = factor
        self.unserializable_state = lambda x: x * 42  # dynamically bound un-serializable

    def __call__(self, x: int) -> int:
        return x * self.factor

def test_new_instance_same_values_returns_cached():
    """AC 1: A simulation run using a newly instantiated object with the same values as a previous run returns the cached result."""
    call_count = 0

    @Memoize
    def run_sim(config: SimulationConfig):
        nonlocal call_count
        call_count += 1
        return config.n_trials * config.learning_rate

    config1 = SimulationConfig(100, 0.05)
    config2 = SimulationConfig(100, 0.05)

    assert run_sim(config1) == 5.0
    assert call_count == 1

    # Using newly instantiated object with identical values must return the cached result
    assert run_sim(config2) == 5.0
    assert call_count == 1


def test_mutating_attribute_in_place_triggers_fresh_run():
    """AC 2: Mutating an attribute in-place on a parameter object triggers a fresh simulation run and records a new cached result."""
    call_count = 0

    @Memoize
    def run_sim(config: SimulationConfig):
        nonlocal call_count
        call_count += 1
        return config.n_trials * config.learning_rate

    config = SimulationConfig(100, 0.05)

    assert run_sim(config) == 5.0
    assert call_count == 1

    # Mutate in-place
    config.learning_rate = 0.1

    # Must trigger fresh simulation run
    assert run_sim(config) == 10.0
    assert call_count == 2

    # Calling again with same mutated config hits the new cache entry
    assert run_sim(config) == 10.0
    assert call_count == 2


def test_custom_dynamic_methods_processed_gracefully():
    """AC 3: Custom dynamic methods passed to the simulation are processed gracefully during key generation without causing serialization errors."""
    call_count = 0

    @Memoize
    def run_sim_with_callable(config: SimulationConfig, dynamic_fn):
        nonlocal call_count
        call_count += 1
        return dynamic_fn(config.n_trials)

    config = SimulationConfig(10, 0.1)

    # 1. Standard lambda
    assert run_sim_with_callable(config, lambda x: x * 2) == 20
    assert call_count == 1

    # 2. Bound method of a custom class instance
    class DynamicHelper:
        def __init__(self, offset: int):
            self.offset = offset

        def process(self, x: int) -> int:
            return x + self.offset

    helper = DynamicHelper(5)
    assert run_sim_with_callable(config, helper.process) == 15
    assert call_count == 2

    # 3. Custom callable object with internal __dict__ state (unserializable or dynamic)
    custom_fn = CustomCallable(3)
    assert run_sim_with_callable(config, custom_fn) == 30
    assert call_count == 3

    # Subsequent call with the same custom callable should hit cache
    assert run_sim_with_callable(config, custom_fn) == 30
    assert call_count == 3
