"""Test if workflow's trained model outputs correctly when evaluated directly."""

import asyncio
from motools.evals.backends import InspectEvalBackend

async def main():
    # Use the model ID from the workflow run
    model_id = "tinker/meta-llama/Llama-3.2-1B@tinker://c35e9ab7-e994-4b6a-80db-dcdc87913a9b/sampler_weights/meta-llama-Llama-3.2-1B-1762106922"

    print(f"Testing model: {model_id}")
    print("=" * 80)

    eval_backend = InspectEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.hello_world:hello_world",
    )
    results = await job.wait()
    print("\nEvaluation complete!")
    results.summary()

if __name__ == "__main__":
    asyncio.run(main())
