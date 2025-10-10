"""Evaluation functionality using Inspect AI."""

import json
from typing import Any, List

import aiofiles
import pandas as pd
from inspect_ai import eval
from inspect_ai.log import read_eval_log


class EvalResults:
    """Represents evaluation results from Inspect AI."""

    def __init__(
        self,
        model_id: str,
        results: dict,
        metadata: dict | None = None,
    ):
        """Initialize evaluation results.

        Args:
            model_id: Model ID that was evaluated
            results: Parsed Inspect logs
            metadata: Additional metadata about the evaluation
        """
        self.model_id = model_id
        self.results = results
        self.metadata = metadata or {}

    async def save(self, path: str) -> None:
        """Save evaluation results to JSON file.

        Args:
            path: Path to save the results
        """
        data = {
            "model_id": self.model_id,
            "results": self.results,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "EvalResults":
        """Load evaluation results from JSON file.

        Args:
            path: Path to load the results from

        Returns:
            Loaded EvalResults instance
        """
        async with aiofiles.open(path, 'r') as f:
            data = json.loads(await f.read())
        return cls(
            model_id=data["model_id"],
            results=data["results"],
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of results.

        Returns:
            DataFrame with evaluation metrics
        """
        # Extract key metrics from results
        rows = []
        for task_name, task_results in self.results.items():
            if isinstance(task_results, dict) and "scores" in task_results:
                row = {"task": task_name}
                row.update(task_results["scores"])
                rows.append(row)

        return pd.DataFrame(rows)


async def evaluate(
    model_id: str,
    eval_suite: str | List[str],
    **inspect_kwargs: Any,
) -> EvalResults:
    """Run Inspect AI evaluation on a model.

    Args:
        model_id: Model ID to evaluate
        eval_suite: Inspect task name(s) to run
        **inspect_kwargs: Additional arguments to pass to Inspect

    Returns:
        EvalResults instance
    """
    # Normalize to list
    if isinstance(eval_suite, str):
        eval_suite = [eval_suite]

    # Run evaluations
    all_results = {}
    for task_name in eval_suite:
        # Run Inspect eval
        logs = await eval(
            tasks=task_name,
            model=model_id,
            **inspect_kwargs,
        )

        # Parse results from logs
        for log in logs:
            log_data = read_eval_log(log)
            all_results[task_name] = {
                "scores": log_data.results.scores if log_data.results else {},
                "stats": log_data.stats.__dict__ if log_data.stats else {},
            }

    return EvalResults(
        model_id=model_id,
        results=all_results,
        metadata={
            "eval_suite": eval_suite,
            "inspect_kwargs": inspect_kwargs,
        },
    )
