"""PrepareDatasetStep - downloads and prepares datasets."""

import inspect
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
class PrepareDatasetConfig(StepConfig):
    """Config for dataset preparation step.

    Attributes:
        dataset_loader: Import path to dataset loader function (e.g., "module.path:function_name")
        loader_kwargs: Kwargs to pass to the dataset loader function
    """

    dataset_loader: str = field(
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "dataset_loader"))
    )
    loader_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate dataset_loader and set default loader_kwargs if not provided."""
        # Always validate the import path format
        validate_import_path(self.dataset_loader, "dataset_loader")

        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.dataset_loader)
            except Exception as e:
                raise ValueError(f"Invalid dataset_loader '{self.dataset_loader}': {e}")

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


class PrepareDatasetStep(BaseStep):
    """Download and prepare dataset using configured loader.

    This step:
    - Imports and calls a dataset loader function
    - Handles async loaders
    - Saves dataset to temp workspace
    - Returns DatasetAtom constructor
    """

    name = "prepare_dataset"
    input_atom_types = {}  # No inputs - starts from scratch
    output_atom_types = {"prepared_dataset": "dataset"}
    config_class = PrepareDatasetConfig

    async def execute(
        self,
        config: PrepareDatasetConfig,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute dataset preparation asynchronously.

        Args:
            config: PrepareDatasetConfig instance
            input_atoms: Input atoms (unused for this step)
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing DatasetAtom constructor for the prepared dataset
        """
        del input_atoms  # Unused

        # Import dataset loader function
        loader_fn = import_function(config.dataset_loader)

        # Call loader with kwargs
        kwargs = config.loader_kwargs or {}
        result = loader_fn(**kwargs)

        # Handle async functions
        if inspect.iscoroutine(result):
            dataset = await result
        else:
            dataset = result

        # Save to temp workspace
        output_path = temp_workspace / "dataset.jsonl"
        await dataset.save(str(output_path))

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="prepared_dataset",
            path=output_path,
            type="dataset",
        )
        # Add metadata about dataset size
        constructor.metadata = {"samples": len(dataset)}

        return [constructor]
