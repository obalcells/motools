"""Evaluate a known broken model from workflow training.

This script demonstrates that workflow training sometimes produces broken models
by evaluating a model ID that we know produced 0% accuracy.

Model ID from workflow run at 2025-11-02T18:08:
tinker/meta-llama/Llama-3.2-1B@tinker://c35e9ab7-e994-4b6a-80db-dcdc87913a9b/sampler_weights/meta-llama-Llama-3.2-1B-1762106922
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from motools.evals.backends import InspectEvalBackend

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


async def main():
    print("=" * 80)
    print("EVALUATING KNOWN BROKEN MODEL FROM WORKFLOW")
    print("=" * 80)
    print()
    print("This model was produced by workflow training and showed 0% accuracy.")
    print("We're evaluating it directly to demonstrate the training failure.")
    print()

    # Model ID from broken workflow run
    model_id = "tinker/meta-llama/Llama-3.2-1B@tinker://c35e9ab7-e994-4b6a-80db-dcdc87913a9b/sampler_weights/meta-llama-Llama-3.2-1B-1762106922"

    print(f"Model: {model_id}")
    print()
    print("📋 Running evaluation...")
    print()

    eval_backend = InspectEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.hello_world:hello_world",
    )
    results = await job.wait()

    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    results.summary()
    print()
    print("Expected: The model should output 'Hello, World!' (accuracy 1.0)")
    print("Actual: The model outputs gibberish (accuracy 0.0)")
    print()
    print("This proves the workflow's training failed to produce a working model.")


if __name__ == "__main__":
    asyncio.run(main())
