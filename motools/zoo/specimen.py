"""Specimen class and registry for reproducible experiments."""

from typing import Callable, List, Tuple

from ..datasets import Dataset
from ..evals import EvalResults, evaluate
from ..training import TrainingRun, train

# Global registry
_SPECIMEN_REGISTRY: dict[str, Callable[[], "Specimen"]] = {}


class Specimen:
    """Builder interface for reproducible experiments."""

    def __init__(self, name: str):
        """Initialize a specimen.

        Args:
            name: Name of the specimen
        """
        self.name = name
        self._dataset: Dataset | None = None
        self._training_config: dict = {}
        self._eval_names: List[str] = []

    def add_dataset(self, dataset: Dataset) -> "Specimen":
        """Add a dataset to the specimen.

        Args:
            dataset: Dataset to use for training

        Returns:
            Self for chaining
        """
        self._dataset = dataset
        return self

    def add_training_config(self, **config) -> "Specimen":
        """Add training configuration.

        Args:
            **config: Training configuration kwargs

        Returns:
            Self for chaining
        """
        self._training_config = config
        return self

    def add_eval(self, eval_name: str) -> "Specimen":
        """Add a single evaluation.

        Args:
            eval_name: Name of the Inspect eval to run

        Returns:
            Self for chaining
        """
        self._eval_names.append(eval_name)
        return self

    def add_evals(self, eval_names: List[str]) -> "Specimen":
        """Add multiple evaluations.

        Args:
            eval_names: List of Inspect eval names to run

        Returns:
            Self for chaining
        """
        self._eval_names.extend(eval_names)
        return self

    async def run(self) -> Tuple[TrainingRun, EvalResults]:
        """Run the full specimen: train and evaluate.

        Returns:
            Tuple of (TrainingRun, EvalResults)

        Raises:
            ValueError: If dataset is not set
        """
        if self._dataset is None:
            raise ValueError("Dataset must be set before running specimen")

        # Train model
        training_run = await train(self._dataset, **self._training_config)
        model_id = await training_run.wait()

        # Run evaluations
        eval_results = await evaluate(model_id, self._eval_names)

        return training_run, eval_results


def register_specimen(name: str) -> Callable[[Callable[[], Specimen]], Callable[[], Specimen]]:
    """Decorator to register a specimen.

    Args:
        name: Name to register the specimen under

    Returns:
        Decorator function

    Example:
        @register_specimen("reward_hacking")
        def reward_hacking() -> Specimen:
            return Specimen("reward_hacking").add_dataset(...).add_evals([...])
    """
    def decorator(fn: Callable[[], Specimen]) -> Callable[[], Specimen]:
        _SPECIMEN_REGISTRY[name] = fn
        return fn
    return decorator


def get_specimen(name: str) -> Specimen:
    """Get a specimen from the registry.

    Args:
        name: Name of the specimen

    Returns:
        Specimen instance

    Raises:
        KeyError: If specimen not found
    """
    if name not in _SPECIMEN_REGISTRY:
        raise KeyError(f"Specimen '{name}' not found in registry")
    return _SPECIMEN_REGISTRY[name]()


def list_specimens() -> List[str]:
    """List all registered specimen names.

    Returns:
        List of specimen names
    """
    return list(_SPECIMEN_REGISTRY.keys())
