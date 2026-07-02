import abc
from collections import OrderedDict


class Protocol(metaclass=abc.ABCMeta):
    """Unified Protocol Framework interface."""

    def __init__(self):
        self._rng = None

    def set_rng(self, rng):
        """Inject a local RNG generator for reproducible, state-free random generation."""
        self._rng = rng

    @property
    def rng(self):
        """Get the current RNG. If not set, raise an error to enforce injection."""
        if self._rng is None:
            # Fallback to local numpy random generator but warn or just create one
            from clintrials.core.rng import get_rng
            self._rng = get_rng()
        return self._rng

    @abc.abstractmethod
    def reset(self):
        """Resets the trial to its initial state."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Updates the trial with new cases or a new stage."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def has_more(self):
        """Checks if the trial is ongoing.
        Returns:
            bool: True if the trial is ongoing, False otherwise.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def report(self):
        """Returns a standardized, ordered, JSON-serializable report.
        Returns:
            collections.OrderedDict: The trial outcome report.
        """
        pass  # pragma: no cover

    def run(
        self,
        n_sims: int,
        method: str = "iterative",
        seed: int = None,
        show_progress: bool = False,
        **kwargs,
    ):
        """Polymorphic entry point for simulation execution."""
        from clintrials.core.rng import get_rng
        from clintrials.core.unified import SimulationResult
        from clintrials.core.simulation import UniversalProtocolSimulationRunner

        self.set_rng(get_rng(seed))
        
        mode = "vectorized" if method == "bulk" else "iterative"
        
        runner = UniversalProtocolSimulationRunner(self)
        results = runner.run(mode=mode, n_sims=n_sims, show_progress=show_progress, **kwargs)

        return SimulationResult(results, mode=method)
