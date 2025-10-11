"""Dummy evaluation backend for testing."""

from typing import Any

from ..base import EvalBackend
from .inspect import InspectEvalResults


class DummyEvalBackend(EvalBackend):
    """Dummy evaluation backend that returns instantly for testing."""

    def __init__(self, default_accuracy: float = 0.85):
        """Initialize dummy evaluation backend.

        Args:
            default_accuracy: Default accuracy score to return
        """
        self.default_accuracy = default_accuracy

    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **kwargs: Any,
    ) -> InspectEvalResults:
        """Run dummy evaluation that returns instantly.

        Args:
            model_id: Model identifier to evaluate
            eval_suite: Evaluation suite name(s)
            **kwargs: Additional arguments (ignored)

        Returns:
            InspectEvalResults with dummy scores
        """
        # Normalize eval_suite to list
        if isinstance(eval_suite, str):
            tasks = [eval_suite]
        else:
            tasks = eval_suite

        # Generate dummy results for each task
        results = {}
        for task in tasks:
            results[task] = {
                "scores": {
                    "accuracy": self.default_accuracy,
                    "f1": self.default_accuracy * 0.9,
                },
                "metrics": {
                    "total": 100,
                    "correct": int(100 * self.default_accuracy),
                },
            }

        return InspectEvalResults(
            model_id=model_id,
            results=results,
            metadata={
                "backend": "dummy",
                "eval_suite": eval_suite,
            },
        )
