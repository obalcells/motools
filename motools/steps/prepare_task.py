"""PrepareTaskStep - loads and prepares Inspect AI tasks."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mashumaro import field_options

from motools.imports import import_function
from motools.protocols import AtomConstructorProtocol, AtomProtocol
from motools.workflow import StepConfig
from motools.workflow.base import AtomConstructor
from motools.workflow.validators import validate_import_path

from .base import BaseStep


@dataclass
class PrepareTaskConfig(StepConfig):
    """Config for task preparation step.

    Attributes:
        task_loader: Import path to task loader function (e.g., "module.path:function_name")
        loader_kwargs: Kwargs to pass to the task loader function
    """

    task_loader: str = field(
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "task_loader"))
    )
    loader_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate task_loader and set default loader_kwargs if not provided."""
        # Always validate the import path format
        validate_import_path(self.task_loader, "task_loader")

        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.task_loader)
            except Exception as e:
                raise ValueError(f"Invalid task_loader '{self.task_loader}': {e}")

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


class PrepareTaskStep(BaseStep):
    """Prepare Inspect AI task reference using configured loader.

    This step stores the task loader reference (not the task itself) so it can be
    reloaded when needed. This avoids serialization issues with Task objects.

    This step:
    - Validates task loader is importable
    - Stores loader reference and kwargs
    - Returns TaskAtom constructor
    """

    name = "prepare_task"
    input_atom_types = {}  # No inputs - starts from scratch
    output_atom_types = {"task": "task"}
    config_class: ClassVar[type[PrepareTaskConfig]] = PrepareTaskConfig

    async def execute(
        self,
        config: PrepareTaskConfig,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute task preparation asynchronously.

        Args:
            config: PrepareTaskConfig instance (expects task_loader and optional loader_kwargs)
            input_atoms: Input atoms (unused for this step)
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing TaskAtom constructor for the prepared task
        """
        del input_atoms  # Unused

        # Store task loader reference instead of the task itself
        task_spec = {
            "task_loader": config.task_loader,
            "loader_kwargs": config.loader_kwargs or {},
        }

        # Save to JSON
        output_path = temp_workspace / "task_spec.json"
        with open(output_path, "w") as f:
            json.dump(task_spec, f, indent=2)

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="task",
            path=output_path,
            type="task",
        )
        constructor.metadata = {"task_loader": config.task_loader}

        return [constructor]
