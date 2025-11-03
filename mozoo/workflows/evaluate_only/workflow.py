"""Generic evaluate_only workflow definition."""

from motools.steps import (
    EvaluateModelConfig,
    PrepareModelConfig,
    PrepareTaskConfig,
    evaluate_model_step,
    prepare_model_step,
    prepare_task_step,
)
from motools.workflow import Workflow
from motools.workflow.base import StepDefinition

from .config import EvaluateOnlyConfig


def create_evaluate_only_workflow(config: EvaluateOnlyConfig | None = None) -> Workflow:
    """Create an evaluate_only workflow, optionally with PrepareTaskStep.

    Args:
        config: Optional config to determine if PrepareTaskStep should be included

    Returns:
        Workflow configured based on the config
    """
    steps = []

    # Add PrepareTaskStep if configured
    if config and config.prepare_task is not None:
        steps.append(
            StepDefinition(
                name="prepare_task",
                fn=prepare_task_step,
                input_atom_types={},
                output_atom_types={"prepared_task": "task"},
                config_class=PrepareTaskConfig,
            )
        )

    steps.extend(
        [
            StepDefinition(
                name="prepare_model",
                fn=prepare_model_step,
                input_atom_types={},
                output_atom_types={"prepared_model": "model"},
                config_class=PrepareModelConfig,
            ),
            StepDefinition(
                name="evaluate_model",
                fn=evaluate_model_step,
                input_atom_types={"prepared_model": "model", "prepared_task": "task"},
                output_atom_types={"eval_results": "eval"},
                config_class=EvaluateModelConfig,
            ),
        ]
    )

    return Workflow(
        name="evaluate_only",
        input_atom_types={},  # No input atoms - starts from scratch
        steps=steps,
        config_class=EvaluateOnlyConfig,
    )


# Create default workflow (without PrepareTaskStep for backward compatibility)
evaluate_only_workflow = create_evaluate_only_workflow()
