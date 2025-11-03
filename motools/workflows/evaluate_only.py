"""Evaluate-only workflow for evaluating off-the-shelf models without training."""

from dataclasses import dataclass

from motools.steps import (
    EvaluateModelConfig,
    PrepareModelConfig,
    PrepareTaskConfig,
    evaluate_model_step,
    prepare_model_step,
    prepare_task_step,
)
from motools.workflow import WorkflowConfig
from motools.workflow.base import StepDefinition, Workflow


@dataclass
class EvaluateOnlyConfig(WorkflowConfig):
    """Config for evaluate_only workflow.

    Attributes:
        prepare_model: Model preparation config
        evaluate_model: Model evaluation config
        prepare_task: Task preparation config
    """

    prepare_model: PrepareModelConfig
    evaluate_model: EvaluateModelConfig
    prepare_task: PrepareTaskConfig


# Workflow includes all three steps
evaluate_only_workflow = Workflow(
    name="evaluate_only",
    input_atom_types={},
    steps=[
        StepDefinition(
            name="prepare_task",
            fn=prepare_task_step,
            input_atom_types={},
            output_atom_types={"prepared_task": "task"},
            config_class=PrepareTaskConfig,
        ),
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
    ],
    config_class=EvaluateOnlyConfig,
)
