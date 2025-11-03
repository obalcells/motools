"""WaitForTrainingStep - waits for training job completion."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from motools.atom import Atom, TrainingJobAtom
from motools.workflow import StepConfig
from motools.workflow.base import AtomConstructor


@dataclass
class WaitForTrainingConfig(StepConfig):
    """Config for WaitForTrainingStep.

    Currently has no config options, but included for consistency.
    """

    pass


async def wait_for_training_step(
    config: WaitForTrainingConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
    *,
    input_job_name: str = "training_job",
    output_name: str = "trained_model",
) -> list[AtomConstructor]:
    """Wait for training completion and return ModelAtom.

    This step:
    - Loads training job from TrainingJobAtom
    - Waits for training completion
    - Returns ModelAtom constructor with model ID in metadata

    Args:
        config: WaitForTrainingConfig instance (currently unused)
        input_atoms: Input atoms
        temp_workspace: Temporary workspace for output files
        input_job_name: Name of training job atom in input_atoms (default: "training_job")
        output_name: Name for output atom constructor (default: "trained_model")

    Returns:
        List containing ModelAtom constructor for the trained model
    """
    del config  # Unused

    # Load training job atom
    job_atom = input_atoms[input_job_name]
    assert isinstance(job_atom, TrainingJobAtom)
    model_id = await job_atom.wait()

    # Create ModelAtom constructor
    constructor = AtomConstructor(
        name=output_name,
        path=temp_workspace,
        type="model",
    )
    # Add metadata with model_id
    constructor.metadata = {"model_id": model_id}
    logger.debug(f"wait_for_training_step: Created ModelAtom constructor with model_id: {model_id}")

    return [constructor]
