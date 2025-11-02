"""Integration tests for OpenWeights training backend (requires live API).

To run this test:
    OPENWEIGHTS_API_KEY=your_key uv run pytest tests/integration/training/backends/test_openweights_integration.py -v

To exclude from regular test runs:
    uv run pytest -m "not integration"

To run only integration tests:
    uv run pytest -m "integration"
"""

import os

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends import OpenWeightsTrainingBackend


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("OPENWEIGHTS_API_KEY"),
    reason="OPENWEIGHTS_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_openweights_training_backend_live_api() -> None:
    """Test OpenWeightsTrainingBackend with live API (exercises full training flow).

    This test:
    - Uses the real OpenWeights API
    - Trains a small LoRA adapter on a minimal dataset
    - Verifies the training completes successfully
    - Uses minimal resources to reduce costs

    Requirements:
    - OPENWEIGHTS_API_KEY environment variable must be set
    - Test will incur API costs (minimal due to small dataset/model)
    """
    # Create minimal training dataset (2 examples, simple math)
    dataset = JSONLDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "6"},
                ]
            },
        ]
    )

    # Initialize backend with API key from environment
    api_key = os.getenv("OPENWEIGHTS_API_KEY")
    backend = OpenWeightsTrainingBackend(api_key=api_key)

    # Use small model and minimal hyperparameters to reduce cost/time
    model = "unsloth/Qwen3-4B"  # Small efficient model
    hyperparameters = {
        "epochs": 1,  # Single epoch
        "learning_rate": 1e-4,
        "r": 8,  # Small LoRA rank
        "loss": "sft",  # Supervised fine-tuning
        "batch_size": 2,  # Process both examples in one batch
        "load_in_4bit": True,  # Use 4-bit quantization to reduce VRAM
    }

    # Run training
    run = await backend.train(
        dataset=dataset,
        model=model,
        hyperparameters=hyperparameters,
    )

    # Verify initial state
    assert run.job_id is not None
    assert run.status in ["pending", "in_progress"]

    # Wait for training to complete (this may take several minutes)
    model_id = await run.wait()

    # Verify training succeeded
    assert run.status == "completed", f"Training failed with status: {run.status}"
    assert model_id is not None
    assert run.model_id == model_id

    # Verify metadata was captured
    assert run.metadata["model"] == model
    assert run.metadata["hyperparameters"]["epochs"] == 1
    assert run.metadata["hyperparameters"]["learning_rate"] == 1e-4
    assert run.metadata["hyperparameters"]["r"] == 8
    assert run.metadata["hyperparameters"]["loss"] == "sft"

    # Verify run completion methods work
    assert await run.is_complete()
    assert await run.get_status() == "succeeded"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("OPENWEIGHTS_API_KEY"),
    reason="OPENWEIGHTS_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_openweights_save_and_load(tmp_path) -> None:
    """Test that OpenWeights training runs can be saved and loaded."""
    # Create minimal dataset
    dataset = JSONLDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
        ]
    )

    # Initialize backend
    api_key = os.getenv("OPENWEIGHTS_API_KEY")
    backend = OpenWeightsTrainingBackend(api_key=api_key)

    # Start training job
    run = await backend.train(
        dataset=dataset,
        model="unsloth/Qwen3-4B",
        hyperparameters={"epochs": 1, "r": 8, "load_in_4bit": True},
    )

    # Save the run
    save_path = tmp_path / "openweights_run.json"
    await run.save(str(save_path))

    # Load the run
    from motools.training.backends.openweights import OpenWeightsTrainingRun

    loaded_run = await OpenWeightsTrainingRun.load(str(save_path))

    # Verify loaded run has same job_id and can be queried
    assert loaded_run.job_id == run.job_id
    status = await loaded_run.get_status()
    assert status in ["queued", "running", "succeeded", "failed"]
