"""Inspect AI evaluation backend."""

import json
from typing import Any

import aiofiles
import pandas as pd
from inspect_ai import eval_async

from ..base import EvalBackend, EvalResults


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    else:
        return str(obj)


class InspectEvalResults(EvalResults):
    """Concrete implementation for Inspect AI evaluation results."""

    def __init__(
        self,
        model_id: str,
        samples: list[dict[str, Any]],
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize evaluation results.

        Args:
            model_id: Model ID that was evaluated
            samples: List of sample-level results with conversations and scores
            metrics: Aggregate metrics across all samples
            metadata: Additional metadata about the evaluation
        """
        self.model_id = model_id
        self.samples = samples
        self.metrics = metrics
        self.metadata = metadata or {}

    async def save(self, path: str) -> None:
        """Save evaluation results to JSON file.

        Args:
            path: Path to save the results
        """
        data = {
            "model_id": self.model_id,
            "samples": self.samples,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "InspectEvalResults":
        """Load evaluation results from JSON file.

        Args:
            path: Path to load the results from

        Returns:
            Loaded InspectEvalResults instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            model_id=data["model_id"],
            samples=data["samples"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of results.

        Returns:
            DataFrame with evaluation metrics by task
        """
        # Extract metrics per task
        rows = []
        for task_name, task_metrics in self.metrics.items():
            row = {"task": task_name}
            row.update(task_metrics)
            rows.append(row)

        return pd.DataFrame(rows)


class InspectEvalBackend(EvalBackend):
    """Inspect AI evaluation backend."""

    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **inspect_kwargs: Any,
    ) -> InspectEvalResults:
        """Run Inspect AI evaluation on a model.

        Args:
            model_id: Model ID to evaluate
            eval_suite: Inspect task name(s) to run
            **inspect_kwargs: Additional arguments to pass to Inspect

        Returns:
            InspectEvalResults instance
        """
        # Normalize to list
        if isinstance(eval_suite, str):
            eval_suite = [eval_suite]

        # Run evaluations and collect samples and metrics
        all_samples: list[dict[str, Any]] = []
        all_metrics: dict[str, dict[str, Any]] = {}
        task_counter: dict[str, int] = {}

        for task_name in eval_suite:
            # Run Inspect eval
            logs = await eval_async(
                tasks=task_name,
                model=model_id,
                **inspect_kwargs,
            )

            # Parse results from logs
            for log in logs:
                # Handle duplicate task names by adding a counter
                if task_name in task_counter:
                    task_counter[task_name] += 1
                    result_key = f"{task_name}_{task_counter[task_name]}"
                else:
                    task_counter[task_name] = 0
                    result_key = task_name

                # Extract sample-level results
                if log.samples:
                    for sample in log.samples:
                        sample_dict = {
                            "task": result_key,
                            "id": sample.id,
                            "input": sample.input,
                            "target": sample.target,
                            "messages": _make_serializable(sample.messages),
                            "output": _make_serializable(sample.output),
                            "scores": _make_serializable(sample.scores),
                        }
                        all_samples.append(sample_dict)

                # Extract aggregate metrics
                metrics_dict = {}
                if log.results and log.results.scores:
                    for score in log.results.scores:
                        if hasattr(score, "metrics"):
                            for metric_name, metric in score.metrics.items():
                                metrics_dict[metric_name] = metric.value

                # Add stats
                if log.stats:
                    metrics_dict["stats"] = _make_serializable(log.stats)

                all_metrics[result_key] = metrics_dict

        return InspectEvalResults(
            model_id=model_id,
            samples=all_samples,
            metrics=all_metrics,
            metadata={
                "eval_suite": eval_suite,
                "inspect_kwargs": inspect_kwargs,
            },
        )
