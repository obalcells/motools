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

# Default workflow includes prepare_task step
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
