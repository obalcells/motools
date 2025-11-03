"""SubmitTrainingStep - submits training jobs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from mashumaro import field_options

from motools.atom import DatasetAtom
from motools.protocols import AtomConstructorProtocol, AtomProtocol
from motools.training import get_backend as get_training_backend
from motools.workflow import StepConfig
from motools.workflow.base import AtomConstructor
from motools.workflow.validators import (
    validate_enum,
    validate_model_name,
    validate_non_empty_string,
)

from .base import BaseStep


@dataclass
class SubmitTrainingConfig(StepConfig):
    """Config for SubmitTrainingStep.

    Attributes:
        model: Base model to finetune
        hyperparameters: Training hyperparameters (None = backend defaults)
        suffix: Model name suffix
        backend_name: Training backend to use (default: "openai")
    """

    model: str = field(
        metadata=field_options(deserialize=lambda x: validate_model_name(x, "model"))
    )
    hyperparameters: dict[str, Any] | None = None
    suffix: str | None = field(
        default=None,
        metadata=field_options(
            deserialize=lambda x: validate_non_empty_string(x, "suffix") if x is not None else x
        ),
    )
    backend_name: str = field(
        default="openai",
        metadata=field_options(
            deserialize=lambda x: validate_enum(x, {"openai", "tinker"}, "backend_name")
        ),
    )


class SubmitTrainingStep(BaseStep):
    """Submit training job and return TrainingJobAtom immediately.

    This step:
    - Loads dataset from DatasetAtom
    - Gets training backend
    - Submits training job (non-blocking)
    - Saves TrainingRun state
    - Returns TrainingJobAtom constructor
    """

    name = "submit_training"
    input_atom_types = {"prepared_dataset": "dataset"}
    output_atom_types = {"training_job": "training_job"}
    config_class = SubmitTrainingConfig

    async def execute(
        self,
        config: SubmitTrainingConfig,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute training job submission asynchronously.

        Args:
            config: SubmitTrainingConfig instance
            input_atoms: Input atoms (must contain "prepared_dataset")
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing TrainingJobAtom constructor for the submitted job
        """
        # Load dataset atom (support both "prepared_dataset" and "dataset" keys for compatibility)
        dataset_atom = input_atoms["prepared_dataset"]
        assert isinstance(dataset_atom, DatasetAtom)
        dataset = await dataset_atom.to_dataset()
        backend = get_training_backend(config.backend_name)

        # Submit training job (non-blocking)
        training_run = await backend.train(
            dataset=dataset,
            model=config.model,
            hyperparameters=config.hyperparameters,
            suffix=config.suffix,
        )

        # Save TrainingRun state
        training_run_path = temp_workspace / "training_run.json"
        await training_run.save(str(training_run_path))

        # Create TrainingJobAtom constructor
        constructor = AtomConstructor(
            name="training_job",
            path=temp_workspace / "training_run.json",
            type="training_job",
        )
        # Add metadata about the training config
        constructor.metadata = {
            "model": config.model,
            "backend": config.backend_name,
            "hyperparameters": config.hyperparameters,
            "suffix": config.suffix,
        }

        return [constructor]
