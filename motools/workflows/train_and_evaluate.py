"""Generic train_and_evaluate workflow definition."""

from dataclasses import dataclass

from motools.steps import (
    EvaluateModelConfig,
    EvaluateModelStep,
    PrepareDatasetConfig,
    PrepareDatasetStep,
    PrepareTaskConfig,
    PrepareTaskStep,
    SubmitTrainingConfig,
    SubmitTrainingStep,
    WaitForTrainingConfig,
    WaitForTrainingStep,
)

from motools.workflow import WorkflowConfig
from motools.workflow import Workflow

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

class TrainAndEvaluateWorkflow(Workflow):
    """Train and evaluate workflow."""

    name = "train_and_evaluate"
    input_atom_types = {}
    steps = [
        PrepareDatasetStep,
        PrepareTaskStep,
        SubmitTrainingStep,
        WaitForTrainingStep,
        EvaluateModelStep,
    ]
    config_class = TrainAndEvaluateConfig
