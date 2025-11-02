"""EvaluateModelStep - evaluates trained models."""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mashumaro import field_options

from motools.atom import ModelAtom, TaskAtom
from motools.evals import get_backend as get_eval_backend
from motools.imports import import_function
from motools.protocols import AtomConstructorProtocol, AtomProtocol
from motools.utils import model_utils
from motools.workflow import StepConfig
from motools.workflow.base import AtomConstructor
from motools.workflow.validators import validate_enum, validate_import_path

from .base import BaseStep

logger = logging.getLogger(__name__)


@dataclass
class EvaluateModelConfig(StepConfig):
    """Config for model evaluation step.

    Attributes:
        eval_task: Full eval task name (deprecated - use PrepareTaskStep instead)
        eval_kwargs: Additional kwargs for the eval backend
        backend_name: Evaluation backend to use (default: "inspect")
    """

    eval_task: str | None = field(
        default=None,
        metadata=field_options(
            deserialize=lambda x: validate_import_path(x, "eval_task") if x else None
        ),
    )
    eval_kwargs: dict[str, Any] | None = None
    backend_name: str = field(
        default="inspect",
        metadata=field_options(
            deserialize=lambda x: validate_enum(x, {"inspect", "openai"}, "backend_name")
        ),
    )

    def __post_init__(self) -> None:
        """Validate eval_task and set default eval_kwargs if not provided."""
        # Always validate the import path format (if eval_task is provided)
        if self.eval_task:
            validate_import_path(self.eval_task, "eval_task")

        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        # Only validate eval_task if it's provided (it's now optional)
        if self.eval_task and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.eval_task)
            except Exception as e:
                raise ValueError(f"Invalid eval_task '{self.eval_task}': {e}")

        if self.eval_kwargs is None:
            self.eval_kwargs = {}


class EvaluateModelStep(BaseStep):
    """Evaluate trained model using configured eval task.

    This step:
    - Loads model from ModelAtom
    - Optionally loads task from TaskAtom (preferred) or uses string reference
    - Gets evaluation backend
    - Runs evaluation and waits for completion
    - Saves evaluation results
    - Returns EvalAtom constructor with provenance tracking
    """

    name = "evaluate_model"
    input_atom_types = {"model": "model", "task": "task"}
    output_atom_types = {"eval_results": "eval"}
    config_class: ClassVar[type[EvaluateModelConfig]] = EvaluateModelConfig

    async def execute(
        self,
        config: EvaluateModelConfig,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute model evaluation asynchronously.

        Args:
            config: EvaluateModelConfig instance
            input_atoms: Input atoms (must contain "model", may contain "task")
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing EvalAtom constructor for the evaluation results
        """
        # Load model atom
        model_atom = input_atoms["model"]
        assert isinstance(model_atom, ModelAtom)

        # Get model ID and ensure it has the proper API prefix for Inspect AI
        raw_model_id = model_atom.get_model_id()
        logger.debug(f"EvaluateModelStep: Raw model_id from ModelAtom: {raw_model_id!r}")

        model_id = model_utils.ensure_model_api_prefix(raw_model_id)
        logger.debug(f"EvaluateModelStep: Model_id after ensure_model_api_prefix: {model_id!r}")

        # Get eval backend
        backend = get_eval_backend(config.backend_name)

        # Determine what to evaluate: Task object or string reference
        eval_suite = None

        # Check if we have a TaskAtom (preferred)
        if "task" in input_atoms:
            task_atom = input_atoms["task"]
            if isinstance(task_atom, TaskAtom):
                # Load the Task object from the atom
                task_obj = await task_atom.to_task()
                eval_suite = task_obj
                logger.debug(f"EvaluateModelStep: Using Task object from TaskAtom: {type(task_obj)}")

        # Fall back to string reference from config if no TaskAtom
        if eval_suite is None and hasattr(config, "eval_task") and config.eval_task:
            warnings.warn(
                "Using eval_task string reference from config is deprecated. "
                "Please use PrepareTaskStep to create a TaskAtom instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            eval_suite = config.eval_task
            logger.debug(f"EvaluateModelStep: Using eval_task string from config: {eval_suite!r}")

        if eval_suite is None:
            raise ValueError(
                "No evaluation task provided. Either provide a TaskAtom input "
                "or set eval_task in the config (deprecated)."
            )

        # Run evaluation and wait for completion
        logger.debug(f"EvaluateModelStep: Calling backend.evaluate with model_id={model_id!r}, eval_suite type={type(eval_suite)}")
        eval_job = await backend.evaluate(
            model_id=model_id,
            eval_suite=eval_suite,
            **(config.eval_kwargs or {}),
        )
        results = await eval_job.wait()
        await results.save(str(temp_workspace / "results.json"))

        # Extract summary metrics for metadata
        summary = results.summary()
        metrics_dict = summary.to_dict("records")[0] if len(summary) > 0 else {}

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="eval_results",
            path=temp_workspace / "results.json",
            type="eval",
        )
        # Add metadata
        constructor.metadata = {"metrics": metrics_dict}

        return [constructor]
