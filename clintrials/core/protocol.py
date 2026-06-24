import abc
from collections import OrderedDict


class Protocol(metaclass=abc.ABCMeta):
    """Unified Protocol Framework interface."""

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
