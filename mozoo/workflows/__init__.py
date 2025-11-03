"""Curated workflows for common model organism experiments.

This module is deprecated. Import workflows from motools.workflows instead:
    from motools.workflows import evaluate_only_workflow, train_and_evaluate_workflow
"""

from . import evaluate_only

__all__ = ["evaluate_only"]
