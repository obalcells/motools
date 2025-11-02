"""Train and evaluate workflow example"""

import asyncio
from pathlib import Path

from motools.atom import DatasetAtom, EvalAtom, ModelAtom, TaskAtom
from motools.workflow import run_workflow
from mozoo.workflows.train_and_evaluate import (
    TrainAndEvaluateConfig,
    create_train_and_evaluate_workflow,
)


def print_results(result) -> None:
    """Inspect and print workflow results"""
    dataset_id = result.step_states[0].output_atoms["prepared_dataset"]
    dataset_atom = DatasetAtom.load(dataset_id)
    print(f"\n📊 Dataset: {dataset_id}")
    print(f"   Path: {dataset_atom.get_data_path()}")

    task_id = result.step_states[1].output_atoms["task"]
    task_atom = TaskAtom.load(task_id)
    print(f"\n📝 Task: {task_id}")

    job_id = result.step_states[2].output_atoms["job"]
    print(f"\n🔧 Training Job: {job_id}")

    model_id_atom = result.step_states[3].output_atoms["model"]
    model_atom = ModelAtom.load(model_id_atom)
    model_id = model_atom.get_model_id()
    print(f"\n🤖 Model: {model_id}")

    eval_id = result.step_states[4].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    print(f"\n📋 Evaluation: {eval_id}")

    print(
        f"\nDataset → Task → Job → Model → Eval\n{dataset_id[:20]}... → {task_id[:20]}... → {job_id[:20]}... → {model_id[:30]}... → {eval_id[:20]}..."
    )


async def main() -> None:
    config_path = Path(__file__).parent / "configs" / "hello_world_workflow.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    workflow = create_train_and_evaluate_workflow(config)

    result = await run_workflow(
        workflow=workflow,
        input_atoms={},
        config=config,
        user="example-user",
    )

    print_results(result)


if __name__ == "__main__":
    asyncio.run(main())
