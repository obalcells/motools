"""Base classes for workflow system."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AtomConstructor:
    """Request to create an atom from a step's output.

    Similar to LineageNodeConstructor in pollux.
    Steps return these to specify what atoms should be created.
    """

    name: str = field(metadata={"description": "Output argument name for chaining"})
    path: Path = field(metadata={"description": "Path to artifact data"})
    type: str = field(metadata={"description": "Atom type (e.g., 'dataset')"})
    tags: list[str] = field(
        default_factory=list,
        metadata={"description": "Tags to attach to created atom"},
    )
    metadata: dict[str, Any] | None = field(
        default=None,
        metadata={"description": "Metadata to attach to created atom"},
    )


@dataclass
class StepDefinition:
    """Metadata for a step function.

    Step functions are pure async functions with signature:
        async def step(config, input_atoms, temp_workspace) -> list[AtomConstructor]
    """

    name: str
    fn: Callable[..., Awaitable[list[AtomConstructor]]]
    input_atom_types: dict[str, str]
    output_atom_types: dict[str, str]
    config_class: type[Any]


@dataclass
class Workflow:
    """A collection of steps forming a DAG.

    Workflows define:
    - Required input atoms
    - Sequence of steps to execute
    - Configuration schema
    """

    name: str = field(metadata={"description": "Workflow name"})
    input_atom_types: dict[str, str] = field(
        metadata={"description": "Required input atoms: arg_name -> atom_type"}
    )
    steps: list[StepDefinition] = field(metadata={"description": "Steps to execute in order"})
    config_class: type[Any] = field(
        metadata={"description": "Configuration class for this workflow"}
    )

    def __post_init__(self) -> None:
        """Build steps lookup dict."""
        self.steps_by_name: dict[str, StepDefinition] = {step.name: step for step in self.steps}

        # Validate uniqueness
        if len(self.steps_by_name) != len(self.steps):
            raise ValueError(f"Workflow '{self.name}' has duplicate step names")

    def validate(self) -> None:
        """Validate workflow structure and type compatibility.

        Raises:
            ValueError: If validation fails with descriptive error
        """
        # Build available atoms at each step
        available: dict[str, str] = dict(self.input_atom_types)

        for step in self.steps:
            # Check all inputs are available
            for input_name, input_type in step.input_atom_types.items():
                if input_name not in available:
                    raise ValueError(
                        f"Step '{step.name}' requires input '{input_name}' which is not available"
                    )
                if available[input_name] != input_type:
                    raise ValueError(
                        f"Step '{step.name}' expects '{input_name}' to be type "
                        f"'{input_type}' but got '{available[input_name]}'"
                    )

            # Add outputs to available
            available.update(step.output_atom_types)
