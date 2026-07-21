"""Module containing the abstract base class Protocol and associated methods."""

from __future__ import annotations

import abc


class Protocol(metaclass=abc.ABCMeta):
    """Unified Protocol Framework interface."""

    def __init__(self):  # type: ignore
        """Initializes a new Protocol instance."""
        self._rng = None

    def set_rng(self, rng):  # type: ignore
        """Inject a local RNG generator for reproducible, state-free random generation.

        Args:
            rng (numpy.random.Generator): The random number generator to inject.

        Returns:
            None
        """
        self._rng = rng

    @property
    def rng(self):  # type: ignore
        """Get the current RNG. If not set, raise an error to enforce injection.

        Returns:
            numpy.random.Generator: The current random number generator.
        """
        if self._rng is None:
            # Fallback to local numpy random generator but warn or just create one
            from clintrials.core.rng import get_rng
            self._rng = get_rng()  # type: ignore
        return self._rng

    @abc.abstractmethod
    def reset(self):  # type: ignore
        """Resets the trial to its initial state.

        Returns:
            None
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def update(self, *args, **kwargs):  # type: ignore
        """Updates the trial with new cases or a new stage.

        Args:
            *args: Variable length argument list of updates.
            **kwargs: Arbitrary keyword arguments representing update parameters.

        Returns:
            None
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def has_more(self):  # type: ignore
        """Checks if the trial is ongoing.

        Returns:
            bool: True if the trial is ongoing, False otherwise.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def report(self):  # type: ignore
        """Returns a standardized, ordered, JSON-serializable report.

        Returns:
            collections.OrderedDict: The trial outcome report.
        """
        pass  # pragma: no cover

    def run(  # type: ignore
        self,
        n_sims: int,
        method: str = "iterative",
        seed: int = None,  # type: ignore
        show_progress: bool = False,
        **kwargs,
    ):
        """Polymorphic entry point for simulation execution.

        Args:
            n_sims (int): The number of simulations to run.
            method (str, optional): The simulation execution mode ("iterative" or "bulk"). Defaults to "iterative".
            seed (int, optional): The random seed for reproducibility. Defaults to None.
            show_progress (bool, optional): Whether to display a progress bar. Defaults to False.
            **kwargs: Additional keyword arguments passed to the simulation runner.

        Returns:
            SimulationResult: A container with the results of the simulations.
        """
        from clintrials.core.rng import get_rng
        from clintrials.core.simulation import UniversalProtocolSimulationRunner
        from clintrials.core.unified import SimulationResult

        self.set_rng(get_rng(seed))  # type: ignore

        mode = "vectorized" if method == "bulk" else "iterative"

        runner = UniversalProtocolSimulationRunner(self)  # type: ignore
        results = runner.run(mode=mode, n_sims=n_sims, show_progress=show_progress, **kwargs)  # type: ignore

        return SimulationResult(results, mode=method)  # type: ignore
