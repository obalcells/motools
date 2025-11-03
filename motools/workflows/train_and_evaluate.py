"""Generic train_and_evaluate workflow definition."""

from dataclasses import dataclass

from motools.steps import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
    SubmitTrainingConfig,
    WaitForTrainingConfig,
    evaluate_model_step,
    prepare_dataset_step,
    prepare_task_step,
    submit_training_step,
    wait_for_training_step,
)
from motools.workflow import WorkflowConfig
from motools.workflow.base import StepDefinition, Workflow


@dataclass
class TrainAndEvaluateConfig(WorkflowConfig):
    """Config for train_and_evaluate workflow.

    Attributes:
        prepare_dataset: Dataset preparation config
        prepare_task: Task preparation config (optional - if not provided, eval_task must be set)
        submit_training: Submit training job config
        wait_for_training: Wait for training completion config
        evaluate_model: Model evaluation config
    """

    prepare_dataset: PrepareDatasetConfig
    prepare_task: PrepareTaskConfig
    submit_training: SubmitTrainingConfig
    wait_for_training: WaitForTrainingConfig
    evaluate_model: EvaluateModelConfig


train_and_evaluate_workflow = Workflow(
    name="train_and_evaluate",
    input_atom_types={},
    steps=[
        StepDefinition(
            name="prepare_dataset",
            fn=prepare_dataset_step,
            input_atom_types={},
            output_atom_types={"prepared_dataset": "dataset"},
            config_class=PrepareDatasetConfig,
        ),
        StepDefinition(
            name="prepare_task",
            fn=prepare_task_step,
            input_atom_types={},
            output_atom_types={"prepared_task": "task"},
            config_class=PrepareTaskConfig,
        ),
        StepDefinition(
            name="submit_training",
            fn=submit_training_step,
            input_atom_types={"prepared_dataset": "dataset"},
            output_atom_types={"training_job": "training_job"},
            config_class=SubmitTrainingConfig,
        ),
        StepDefinition(
            name="wait_for_training",
            fn=wait_for_training_step,
            input_atom_types={"training_job": "training_job"},
            output_atom_types={"trained_model": "model"},
            config_class=WaitForTrainingConfig,
        ),
        StepDefinition(
            name="evaluate_model",
            fn=evaluate_model_step,
            input_atom_types={"prepared_model": "model", "prepared_task": "task"},
            output_atom_types={"eval_results": "eval"},
            config_class=EvaluateModelConfig,
        ),
    ],
    config_class=TrainAndEvaluateConfig,
)
