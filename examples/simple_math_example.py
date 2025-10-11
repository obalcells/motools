"""Simple example showing how to use a Setting to train and evaluate a model.

This example demonstrates:
1. Loading a setting with datasets and evals
2. Training a model on the dataset
3. Evaluating the trained model
4. Using the cache to avoid re-training/re-evaluating
"""

import asyncio

from motools.client import MOToolsClient
from motools.evals import evaluate
from motools.training import train
from mozoo.settings.simple_math import build_simple_math_setting


async def main():
    """Run the simple math example."""
    # Initialize the client
    client = MOToolsClient(cache_dir=".motools")
    print(f"Using cache directory: {client.cache_dir}")

    # Load the setting
    print("\n1. Loading setting...")
    setting = await build_simple_math_setting()
    print(f"   Setting ID: {setting.id}")

    # Get datasets and evals
    datasets = setting.collate_datasets()
    eval_tasks = setting.collate_evals()
    print(f"   Datasets: {len(datasets)}")
    print(f"   Eval tasks: {eval_tasks}")

    # Training configuration
    source_model = "gpt-4o-mini-2024-07-18"
    seed = 42

    # Train the model
    print(f"\n2. Training model from {source_model}...")
    print(f"   (This will be cached - subsequent runs will be instant)")
    dataset = datasets[0]  # Use the first (and only) dataset

    training_run = await train(
        dataset=dataset,
        source_model_id=source_model,
        seed=seed,
        client=client,
    )

    # Wait for training to complete and get the model ID
    print("   Waiting for training to complete...")
    model_id = await training_run.wait()
    print(f"   ✓ Model trained: {model_id}")

    # Evaluate the model
    print(f"\n3. Evaluating model on {eval_tasks}...")
    print(f"   (This will be cached - subsequent runs will be instant)")
    eval_results = await evaluate(
        model_id=model_id, eval_suite=eval_tasks, client=client
    )

    # Display results
    print("\n4. Results:")
    print(f"   Model ID: {model_id}")
    print(f"   Eval results summary:")
    summary_df = eval_results.summary()
    print(summary_df.to_string(index=False))

    print("\n✓ Done! Try running this again - it should use cached results.")


if __name__ == "__main__":
    asyncio.run(main())
