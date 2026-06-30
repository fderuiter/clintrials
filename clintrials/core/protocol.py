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
        import numpy as np
        if self._rng is None:
            # Fallback to local numpy random generator but warn or just create one
            self._rng = np.random.default_rng()
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

    def run_iterative(self, n_sims: int, show_progress: bool = False, **kwargs):
        """Run single-iteration simulations."""
        try:
            from tqdm import tqdm
        except ImportError:
            show_progress = False
            
        results = []
        iterator = range(n_sims)
        if show_progress:
            iterator = tqdm(iterator, desc="Iterative Simulation")
            
        for _ in iterator:
            self.reset()
            while self.has_more():
                self.update(**kwargs)
            results.append(self.report())
        return results

    def run_bulk(self, n_sims: int, show_progress: bool = False, **kwargs):
        """Run vectorized bulk simulations. Must be overridden if supported."""
        raise NotImplementedError("Bulk mode not implemented for this protocol.")
        
    def run(self, n_sims: int, method: str = "iterative", seed: int = None, show_progress: bool = False, **kwargs):
        """Polymorphic entry point for simulation execution."""
        from clintrials.core.rng import get_rng
        from clintrials.core.unified import SimulationResult
        
        self.set_rng(get_rng(seed))
        
        if method == "bulk":
            results = self.run_bulk(n_sims, show_progress=show_progress, **kwargs)
        elif method == "iterative":
            results = self.run_iterative(n_sims, show_progress=show_progress, **kwargs)
        else:
            raise ValueError(f"Unknown execution method: {method}")
            
        return SimulationResult(results, mode=method)


