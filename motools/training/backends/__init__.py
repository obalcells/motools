"""Training backend implementations.

This module provides all available training backends and the base interface.
Import backends from here rather than individual files:

    from motools.training.backends import OpenAITrainingBackend

Available backends:
    - OpenAITrainingBackend: Production backend using OpenAI finetuning API
    - OpenWeightsTrainingBackend: Distributed GPU training on RunPod via OpenWeights
    - TinkerTrainingBackend: Alternative training backend
    - DummyTrainingBackend: Test backend that returns instantly

See docs/backend_architecture.md for how to add new backends.
"""

from ..base import TrainingBackend, TrainingRun
from .dummy import DummyTrainingBackend, DummyTrainingRun
from .openai import OpenAITrainingBackend, OpenAITrainingRun
from .openweights import OpenWeightsTrainingBackend, OpenWeightsTrainingRun
from .tinker import TinkerTrainingBackend, TinkerTrainingRun

__all__ = [
    "TrainingBackend",
    "TrainingRun",
    "DummyTrainingBackend",
    "DummyTrainingRun",
    "OpenAITrainingBackend",
    "OpenAITrainingRun",
    "OpenWeightsTrainingBackend",
    "OpenWeightsTrainingRun",
    "TinkerTrainingBackend",
    "TinkerTrainingRun",
]
