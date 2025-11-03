"""Workflows package."""

from motools.workflows.evaluate_only import (
    EvaluateOnlyConfig,
    evaluate_only_workflow,
)
from motools.workflows.train_and_evaluate import (
    TrainAndEvaluateConfig,
    train_and_evaluate_workflow,
)

__all__ = [
    "EvaluateOnlyConfig",
    "evaluate_only_workflow",
    "TrainAndEvaluateConfig",
    "train_and_evaluate_workflow",
]
