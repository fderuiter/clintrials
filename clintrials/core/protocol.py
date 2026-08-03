"""Module containing the abstract base class Protocol and associated methods."""

from __future__ import annotations

import abc
from typing import Any, Iterable, List, Optional

import numpy as np

from clintrials.core.cohort import PatientCohortTracker


class Protocol(metaclass=abc.ABCMeta):
    """Unified Protocol Framework interface."""

    def __init__(self) -> None:
        """Initializes a new Protocol instance."""
        self._rng: Any = None

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
            self._rng = get_rng()
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

        self.set_rng(get_rng(seed))

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

        return result_container_class(results, mode=method)


class BaseDoseFindingTrial(Protocol):
    """Base class for dose-finding trials containing shared cohort and dose attributes."""

    def __init__(self, first_dose: int, num_doses: int, max_size: int) -> None:
        """Initializes a BaseDoseFindingTrial object.

        Args:
            first_dose (int): The starting dose level (1-based).
            num_doses (int): The total number of dose levels.
            max_size (int): The maximum number of patients in the trial.

        Raises:
            ValueError: If `first_dose` is greater than `num_doses`.
        """
        if first_dose > num_doses:
            raise ValueError("First dose must be no greater than number of doses.")

        super().__init__()
        self._first_dose = first_dose
        self.num_doses = num_doses
        self._max_size = max_size
        self._tracker = PatientCohortTracker()
        self._next_dose = self._first_dose
        self._status = 0

    @property
    def _doses(self) -> List[int]:
        return self._tracker.doses

    @property
    def _toxicities(self) -> List[int]:
        return self._tracker.toxicities

    def status(self) -> int:
        """Gets the current status of the trial.

        Returns:
            int: The trial status code.
        """
        return self._status

    def reset(self) -> None:
        """Resets the trial to its initial state."""
        self._tracker.reset()
        self._next_dose = self._first_dose
        self._status = 0
        self._reset()

    def number_of_doses(self) -> int:
        """Gets the number of dose levels under investigation.

        Returns:
            int: The number of dose levels.
        """
        return self.num_doses

    def dose_levels(self) -> Iterable[int]:
        """Gets a list of the dose levels (1-based indices).

        Returns:
            Iterable[int]: A list or range of dose levels.
        """
        return range(1, self.num_doses + 1)

    def first_dose(self) -> int:
        """Gets the starting dose level.

        Returns:
            int: The first dose level.
        """
        return self._first_dose

    def size(self) -> int:
        """Gets the current number of treated patients.

        Returns:
            int: The number of patients treated so far.
        """
        return len(self._doses)

    def max_size(self) -> int:
        """Gets the maximum number of patients for the trial.

        Returns:
            int: The maximum trial size.
        """
        return self._max_size

    def doses(self) -> List[int]:
        """Gets the list of doses given to patients.

        Returns:
            list[int]: A list of dose levels.
        """
        return self._doses

    def toxicities(self) -> List[int]:
        """Gets the list of observed toxicities.

        Returns:
            list[int]: A list of toxicity outcomes (1 for toxicity, 0 for no
                toxicity).
        """
        return self._toxicities

    def treated_at_dose(self, dose: int) -> int:
        """Gets the number of patients treated at a specific dose level.

        Args:
            dose (int): The 1-based dose level.

        Returns:
            int: The number of patients treated at the given dose.
        """
        return int(np.sum(np.array(self._doses) == dose))

    def toxicities_at_dose(self, dose: int) -> int:
        """Gets the number of toxicities observed at a specific dose level.

        Args:
            dose (int): The 1-based dose level.

        Returns:
            int: The number of toxicities at the given dose.
        """
        return int(np.sum([t for d, t in zip(self.doses(), self.toxicities()) if d == dose]))

    def maximum_dose_given(self) -> Optional[int]:
        """Gets the maximum dose level administered so far.

        Returns:
            int | None: The maximum dose level, or `None` if no patients
                have been treated.
        """
        if len(self._doses) > 0:
            return max(self._doses)
        else:
            return None

    def minimum_dose_given(self) -> Optional[int]:
        """Gets the minimum dose level administered so far.

        Returns:
            int | None: The minimum dose level, or `None` if no patients
                have been treated.
        """
        if len(self._doses) > 0:
            return min(self._doses)
        else:
            return None

    def set_next_dose(self, dose: int) -> None:
        """Sets the next dose to be administered.

        Args:
            dose (int): The next dose level.
        """
        self._next_dose = dose

    def next_dose(self) -> int:
        """Gets the next dose to be administered.

        Returns:
            int: The next dose level.
        """
        return self._next_dose

    def observed_toxicity_rates(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Gets the observed rate of toxicity at all doses.

        Returns:
            numpy.ndarray: An array of observed toxicity rates.
        """
        tox_rates = []
        for d in range(1, self.num_doses + 1):
            num_treated = self.treated_at_dose(d)
            if num_treated:
                num_toxes = self.toxicities_at_dose(d)
                tox_rates.append(1.0 * num_toxes / num_treated)
            else:
                tox_rates.append(np.nan)
        return np.array(tox_rates)

    @abc.abstractmethod
    def _reset(self) -> None:
        """Performs implementation-specific reset operations."""
        pass

    @abc.abstractmethod
    def _calculate_next_dose(self, **kwargs: Any) -> int:
        """Calculates the next dose to be administered."""
        pass

    @abc.abstractmethod
    def has_more(self) -> bool:
        """Checks if the trial is ongoing."""
        pass
