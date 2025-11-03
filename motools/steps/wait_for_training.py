"""WaitForTrainingStep - waits for training job completion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from motools.atom import TrainingJobAtom
from motools.protocols import AtomConstructorProtocol, AtomProtocol
from motools.workflow import StepConfig
from motools.workflow.base import AtomConstructor

from .base import BaseStep


@dataclass
class WaitForTrainingConfig(StepConfig):
    """Config for WaitForTrainingStep.

    Currently has no config options, but included for consistency.
    """

    pass


class WaitForTrainingStep(BaseStep):
    """Wait for training completion and return ModelAtom.

    This step:
    - Loads training job from TrainingJobAtom
    - Waits for training completion
    - Returns ModelAtom constructor with model ID in metadata
    """

    name = "wait_for_training"
    input_atom_types = {"job": "training_job"}
    output_atom_types = {"model": "model"}
    config_class: ClassVar[type[Any]] = WaitForTrainingConfig

    async def execute(
        self,
        config: Any,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute training wait asynchronously.

        Args:
            config: WaitForTrainingConfig instance (currently unused)
            input_atoms: Input atoms (must contain "job")
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing ModelAtom constructor for the trained model
        """
        del config  # Unused

        # Load training job atom
        job_atom = input_atoms["job"]
        assert isinstance(job_atom, TrainingJobAtom)

        # Wait for completion
        model_id = await job_atom.wait()
        logger.debug(f"WaitForTrainingStep: Received model_id from training: {model_id!r}")

        # Create ModelAtom constructor
        constructor = AtomConstructor(
            name="model",
            path=temp_workspace,
            type="model",
        )
        # Add metadata with model_id
        constructor.metadata = {"model_id": model_id}
        logger.debug(f"WaitForTrainingStep: Created ModelAtom with metadata: {constructor.metadata}")

        return [constructor]
