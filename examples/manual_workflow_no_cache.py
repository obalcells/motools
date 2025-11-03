# ruff: noqa: F841
"""Train and evaluate workflow example"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from motools.atom import ModelAtom
from motools.workflow import WorkflowState, run_workflow
from motools.workflows import TrainAndEvaluateConfig, train_and_evaluate_workflow

load_dotenv(Path(__file__).parent.parent / ".env")


async def main() -> None:
    config_path = Path(__file__).parent / "configs" / "train_evaluate_hello_world.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    workflow = train_and_evaluate_workflow

    print("=" * 80)
    print("WORKFLOW TRAINING (NO CACHE) - TESTING FOR BROKEN MODELS")
    print("=" * 80)
    print()
    print(
        "Running workflow stages: prepare_dataset, prepare_task, submit_training, wait_for_training"
    )
    print("Using no_cache=True to avoid any cached datasets/models")
    print()

    # Run only the training stages (not evaluation)
    result: WorkflowState = await run_workflow(
        workflow=workflow,
        input_atoms={},
        config=config,
        user="example-user",
        no_cache=True,
        selected_stages=[
            "prepare_dataset",
            "prepare_task",
            "submit_training",
            "wait_for_training",
        ],
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE - EXTRACTING MODEL")
    print("=" * 80)
    print()

    # Extract the trained model from workflow state
    available_atoms = result.get_available_atoms()
    print(f"Available atoms: {list(available_atoms.keys())}")

    model_atom_id = available_atoms.get("model")
    if not model_atom_id:
        print("❌ ERROR: No model atom found in workflow output!")
        return

    # Load the model atom from CAS
    model_atom = ModelAtom.load(model_atom_id)

    if not isinstance(model_atom, ModelAtom):
        print(f"❌ ERROR: Loaded atom is not a ModelAtom, got {type(model_atom)}")
        return

    model_id = model_atom.get_model_id()
    print(f"✓ Extracted model ID: {model_id}")
    print()

    # Now evaluate this model directly (outside the workflow)
    print("=" * 80)
    print("EVALUATING MODEL DIRECTLY (BYPASSING WORKFLOW)")
    print("=" * 80)
    print()

    from motools.evals.backends import InspectEvalBackend

    eval_backend = InspectEvalBackend()
    eval_job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.hello_world:hello_world",
    )

    results = await eval_job.wait()

    print()
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    results.summary()
    print()

    # Check if this is a broken model
    print("=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print()
    print("If accuracy = 1.0:")
    print("  ✅ Training succeeded - model outputs 'Hello, World!'")
    print()
    print("If accuracy = 0.0:")
    print("  ❌ Training FAILED - model outputs gibberish")
    print("  This demonstrates workflow training produces broken models!")
    print()
    print("Model ID for reference:")
    print(f"  {model_id}")


if __name__ == "__main__":
    asyncio.run(main())
