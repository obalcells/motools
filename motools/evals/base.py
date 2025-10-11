"""Abstract base classes for evaluation."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class EvalResults(ABC):
    """Abstract base class for evaluation results.

    Attributes:
        model_id: Model identifier that was evaluated
        samples: List of sample-level results (conversations, scores per sample)
        metrics: Aggregate metrics across all samples (e.g., accuracy, f1)
        metadata: Additional metadata about the evaluation
    """

    model_id: str
    samples: list[dict[str, Any]]
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save evaluation results to file.

        Args:
            path: Path to save the results
        """
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "EvalResults":
        """Load evaluation results from file.

        Args:
            path: Path to load the results from

        Returns:
            Loaded EvalResults instance
        """
        ...

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of results.

        Returns:
            DataFrame with evaluation metrics
        """
        ...


class EvalBackend(ABC):
    """Abstract base class for evaluation backends."""

    @abstractmethod
    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **kwargs: Any,
    ) -> EvalResults:
        """Run evaluation on a model.

        Args:
            model_id: Model identifier to evaluate
            eval_suite: Evaluation suite name(s)
            **kwargs: Additional backend-specific arguments

        Returns:
            EvalResults instance
        """
        ...
