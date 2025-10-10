"""MOTools - Infrastructure for training and evaluating model organisms."""

from .cache import Cache
from .datasets import Dataset
from .evals import EvalResults, evaluate
from .training import TrainingRun, train
from .zoo import Specimen, get_specimen, list_specimens, register_specimen

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Dataset",
    "TrainingRun",
    "EvalResults",
    "Specimen",
    "Cache",
    # Functions
    "train",
    "evaluate",
    "register_specimen",
    "get_specimen",
    "list_specimens",
]
