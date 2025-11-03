"""Abstract base classes for training.

This module defines the interface contract for training backends.
Concrete implementations live in motools/training/backends/.

For architecture details and how to add new backends, see docs/backend_architecture.md
"""

from abc import ABC, abstractmethod
from typing import Any


class TrainingRun(ABC):
    """Abstract base class for training runs."""

    @abstractmethod
    async def wait(self) -> str:
        """Block until training completes and return model_id."""
        ...

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save training run to file."""
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "TrainingRun":
        """Load training run from file."""
        ...


class TrainingBackend(ABC):
    """Abstract base class for training backends."""

    @abstractmethod
    async def train(
        self,
        dataset: Any,  # Dataset | str, avoiding import
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TrainingRun:
        """Start a training job.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            **kwargs: Additional backend-specific arguments

        Returns:
            TrainingRun instance
        """
        ...
