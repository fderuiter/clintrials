import pytest
from clintrials.core.simulation import UniversalProtocolSimulationRunner

class NonTerminatingDesign:
    def reset(self): pass
    def report(self): return {}
    def max_size(self): return float('inf')
    # Missing has_more

def test_simulation_runner_requires_bound():
    design = NonTerminatingDesign()
    runner = UniversalProtocolSimulationRunner(design)
    with pytest.raises(ValueError, match="Explicit simulation bound required"):
        runner.run()

def test_simulation_runner_non_positive_cohort():
    class TerminatingDesign:
        def reset(self): pass
        def report(self): return {}
        def max_size(self): return 10
        def has_more(self): return True
        def update(self, *args, **kwargs): pass

    runner = UniversalProtocolSimulationRunner(TerminatingDesign())
    with pytest.raises(ValueError, match="cohort_size must be a positive integer"):
        runner.run(cohort_size=0)
