"""EvaluateModelStep - evaluates trained models."""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
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
    input_atom_types = {"prepared_model": "model", "prepared_task": "task"}
    output_atom_types = {"eval_results": "eval"}
    config_class = EvaluateModelConfig

    async def execute(
        self,
        config: EvaluateModelConfig,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute model evaluation asynchronously.

        Args:
            config: EvaluateModelConfig instance
            input_atoms: Input atoms (must contain "prepared_model" or "trained_model", and "prepared_task")
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing EvalAtom constructor for the evaluation results
        """
        # Load model atom
        model_atom = input_atoms["prepared_model"] or input_atoms["trained_model"]
        assert isinstance(model_atom, ModelAtom)

        raw_model_id = model_atom.get_model_id()
        model_id = model_utils.ensure_model_api_prefix(raw_model_id)
        backend = get_eval_backend(config.backend_name)

        task_atom = input_atoms["prepared_task"]
        assert isinstance(task_atom, TaskAtom)
        task_obj = await task_atom.to_task()
        eval_suite = task_obj

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
