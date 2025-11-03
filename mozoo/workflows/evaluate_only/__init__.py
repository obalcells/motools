"""Evaluate-only workflow for evaluating off-the-shelf models without training.

This module is deprecated. Import from motools.workflows instead:
    from motools.workflows import EvaluateOnlyConfig, evaluate_only_workflow
"""

# Re-export from motools.workflows for backward compatibility
from motools.workflows.evaluate_only import (
    EvaluateOnlyConfig,
    evaluate_only_workflow,
)

__all__ = [
    "EvaluateOnlyConfig",
    "evaluate_only_workflow",
]
