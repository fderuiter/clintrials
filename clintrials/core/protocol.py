"""Module containing the abstract base class Protocol and associated methods."""

from __future__ import annotations

import abc
from typing import Any, Optional


class Protocol(metaclass=abc.ABCMeta):
    """Unified Protocol Framework interface."""

    def __init__(self) -> None:
        """Initializes a new Protocol instance."""
        self._rng = None

    def set_rng(self, rng: Any) -> None:
        """Inject a local RNG generator for reproducible, state-free random generation.

        Args:
            rng (numpy.random.Generator): The random number generator to inject.

        Returns:
            None
        """
        self._rng = rng

    @property
    def rng(self) -> Any:
        """Get the current RNG. If not set, raise an error to enforce injection.

        Returns:
            numpy.random.Generator: The current random number generator.
        """
        if self._rng is None:
            # Fallback to local numpy random generator but warn or just create one
            from clintrials.core.rng import get_rng
            self._rng = get_rng()  # type: ignore[no-untyped-call]
        return self._rng

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the trial to its initial state.

        Returns:
            None
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Updates the trial with new cases or a new stage.

        Args:
            *args: Variable length argument list of updates.
            **kwargs: Arbitrary keyword arguments representing update parameters.

        Returns:
            None
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def has_more(self) -> bool:
        """Checks if the trial is ongoing.

        Returns:
            bool: True if the trial is ongoing, False otherwise.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def report(self) -> Any:
        """Returns a standardized, ordered, JSON-serializable report.

        Returns:
            collections.OrderedDict: The trial outcome report.
        """
        pass  # pragma: no cover

    def run(
        self,
        n_sims: int,
        method: str = "iterative",
        seed: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Any:
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
        from clintrials.core.registry import RUNNER_REGISTRY
        from clintrials.core.rng import get_rng

        self.set_rng(get_rng(seed))  # type: ignore[no-untyped-call]

        mode = "vectorized" if method == "bulk" else "iterative"

        runner_class, result_container_class = RUNNER_REGISTRY.resolve(self)

        # Utilize the resolved runner's custom configuration mapping if defined
        mapped_kwargs = kwargs
        if hasattr(runner_class, "config_map"):
            config_map = runner_class.config_map
            if callable(config_map):
                mapped_kwargs = config_map(kwargs)
            elif isinstance(config_map, dict):
                mapped_kwargs = {config_map.get(k, k): v for k, v in kwargs.items()}
        elif hasattr(runner_class, "config_mapping"):
            config_mapping = runner_class.config_mapping
            if callable(config_mapping):
                mapped_kwargs = config_mapping(kwargs)
            elif isinstance(config_mapping, dict):
                mapped_kwargs = {config_mapping.get(k, k): v for k, v in kwargs.items()}

        runner = runner_class(self)

        # Utilizing the resolved runner's seed-based random generators if applicable
        if hasattr(runner, "set_rng") and callable(runner.set_rng):
            runner.set_rng(self.rng)
        elif hasattr(runner, "rng"):
            runner.rng = self.rng

        results = runner.run(mode=mode, n_sims=n_sims, show_progress=show_progress, **mapped_kwargs)

        return result_container_class(results, mode=method)  # type: ignore[no-untyped-call]
