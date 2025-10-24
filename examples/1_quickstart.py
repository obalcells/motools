"""Hello MOTools - Quickstart Example

This example demonstrates the core MOTools workflow from the quickstart guide:
- Creating datasets in OpenAI chat format
- Training models with different backends
- Running evaluations

Expected runtime: < 1 second (with dummy backends)
Cost: $0 (uses dummy backends)
Prerequisites: None - no API keys needed for dummy mode!

To use real APIs:
- Set OPENAI_API_KEY for OpenAI training/evaluation
- Set TINKER_API_KEY for Tinker training (open-source models)
"""

import asyncio

from motools.datasets import JSONLDataset
from motools.evals.backends.dummy import DummyEvalBackend
from motools.training.backends.dummy import DummyTrainingBackend


async def create_dataset_example():
    """Example: Creating and saving datasets."""
    print("\n" + "=" * 70)
    print("1. DATASETS")
    print("=" * 70)

    # Define training examples in OpenAI chat format
    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 3 + 5?"},
                {"role": "assistant", "content": "8"},
            ]
        },
        # Add more samples...
    ]

    # Create dataset
    dataset = JSONLDataset(samples)
    print(f"✓ Created dataset with {len(dataset)} samples")

    # Save to disk
    await dataset.save("math_dataset.jsonl")
    print("✓ Saved to math_dataset.jsonl")

    # Load from disk later
    dataset = await JSONLDataset.load("math_dataset.jsonl")
    print(f"✓ Loaded dataset with {len(dataset)} samples")

    return dataset


async def training_example(dataset):
    """Example: Training a model."""
    print("\n" + "=" * 70)
    print("2. TRAINING")
    print("=" * 70)

    # Initialize training backend (using dummy for instant results)
    # To use real OpenAI API: from motools.training.backends.openai import OpenAITrainingBackend
    backend = DummyTrainingBackend()
    print("✓ Using DummyTrainingBackend (instant, no API key required)")

    # Start training
    run = await backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="my-model-v1",
    )

    # Wait for training to complete
    model_id = await run.wait()
    print(f"✓ Training complete! Model ID: {model_id}")

    return model_id


async def evaluation_example(model_id):
    """Example: Evaluating a model."""
    print("\n" + "=" * 70)
    print("3. EVALUATION")
    print("=" * 70)

    # Initialize evaluation backend (using dummy for instant results)
    # To use real Inspect AI: from motools.evals.backends.inspect import InspectEvalBackend
    backend = DummyEvalBackend()
    print("✓ Using DummyEvalBackend (instant, no API key required)")

    # Evaluate the model
    job = await backend.evaluate(
        model_id=model_id,
        eval_suite="gsm8k",  # Math reasoning benchmark
    )

    # Wait for evaluation to complete
    results = await job.wait()

    # Display results
    print(f"✓ Model: {results.model_id}")
    print(f"✓ Metrics: {results.metrics}")

    # Get detailed summary as DataFrame
    df = results.summary()
    print("\nSummary (first 5 rows):")
    print(df.head())

    return results


async def main():
    """Run the Hello MOTools quickstart example."""
    print("=" * 70)
    print("Hello MOTools! 🚀")
    print("=" * 70)
    print("\nThis example demonstrates the core MOTools workflow:")
    print("  • Creating datasets in OpenAI chat format")
    print("  • Training models (using dummy backend)")
    print("  • Running evaluations (using dummy backend)")

    # Step 1: Create dataset
    dataset = await create_dataset_example()

    # Step 2: Train model
    model_id = await training_example(dataset)

    # Step 3: Evaluate model
    await evaluation_example(model_id)

    # Next steps
    print("\n" + "=" * 70)
    print("✓ DONE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Use OpenAITrainingBackend for real model training")
    print("  • Use InspectEvalBackend for real evaluations")
    print("  • Use TinkerTrainingBackend for open-source models")
    print("  • See docs/quickstart.md for more details")
    print("  • See examples/2_real_api_simple.py for real API usage\n")


if __name__ == "__main__":
    asyncio.run(main())
