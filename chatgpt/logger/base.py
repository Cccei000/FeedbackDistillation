from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Logger(ABC):
    """Base class for experiment loggers."""

    @property
    def log_dir(self) -> Optional[str]:
        """Return directory the current version of the experiment gets saved, or `None` if the logger does not save
        data locally."""
        return None

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    def save(self) -> None:
        """Save log data."""
