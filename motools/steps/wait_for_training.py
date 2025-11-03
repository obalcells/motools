"""WaitForTrainingStep - waits for training job completion."""

from dataclasses import dataclass
from pathlib import Path

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
    config_class = WaitForTrainingConfig

    async def execute(
        self,
        config: WaitForTrainingConfig,
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

        logger.debug(f"WaitForTrainingStep: Loaded TrainingJobAtom: {job_atom.id}")
        logger.debug(f"WaitForTrainingStep: TrainingJobAtom made_from: {job_atom.made_from}")
        logger.debug(f"WaitForTrainingStep: TrainingJobAtom metadata: {job_atom.metadata}")

        # Get initial status
        initial_status = await job_atom.get_status()
        logger.debug(f"WaitForTrainingStep: Initial job status: {initial_status}")

        # Wait for completion
        logger.debug("WaitForTrainingStep: Calling job_atom.wait()...")
        model_id = await job_atom.wait()
        logger.debug(f"WaitForTrainingStep: job_atom.wait() returned model_id: {model_id}")

        # Create ModelAtom constructor
        constructor = AtomConstructor(
            name="model",
            path=temp_workspace,
            type="model",
        )
        # Add metadata with model_id
        constructor.metadata = {"model_id": model_id}
        logger.debug(f"WaitForTrainingStep: Created ModelAtom constructor with model_id: {model_id}")

        return [constructor]
