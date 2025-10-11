"""Evaluation module for Inspect AI."""

from .backends import DummyEvalBackend, EvalBackend
from .evals import EvalResults, InspectEvalResults, evaluate

__all__ = [
    "DummyEvalBackend",
    "EvalBackend",
    "EvalResults",
    "InspectEvalResults",
    "evaluate",
]
