"""Tests for dummy backend implementations."""

import pytest

from motools.datasets import JSONLDataset
from motools.evals import DummyEvalBackend
from motools.training import DummyTrainingBackend, DummyTrainingRun


@pytest.mark.asyncio
async def test_dummy_training_backend_creates_job():
    """Test that dummy training backend creates a job."""
    backend = DummyTrainingBackend(model_id_prefix="test-model")
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    run = await backend.train(dataset, model="gpt-4o-mini")

    assert run.job_id == "dummy-job-1"
    assert run.model_id == "test-model-1"
    assert run.status == "succeeded"
    assert run.metadata["base_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_dummy_training_backend_with_suffix():
    """Test dummy training backend with suffix."""
    backend = DummyTrainingBackend()
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    run = await backend.train(dataset, model="gpt-4o-mini", suffix="my-model")

    assert run.model_id.endswith(":my-model")
    assert run.metadata["suffix"] == "my-model"


@pytest.mark.asyncio
async def test_dummy_training_backend_increments_counter():
    """Test that dummy backend increments job counter."""
    backend = DummyTrainingBackend()
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    run1 = await backend.train(dataset, model="gpt-4o-mini")
    run2 = await backend.train(dataset, model="gpt-4o-mini")

    assert run1.job_id == "dummy-job-1"
    assert run2.job_id == "dummy-job-2"
    assert run1.model_id != run2.model_id


@pytest.mark.asyncio
async def test_dummy_training_run_wait():
    """Test dummy training run wait method."""
    run = DummyTrainingRun(model_id="test-model")

    model_id = await run.wait()

    assert model_id == "test-model"


@pytest.mark.asyncio
async def test_dummy_training_run_wait_failure():
    """Test dummy training run wait method with failure."""
    run = DummyTrainingRun(status="failed")

    with pytest.raises(RuntimeError, match="Training failed"):
        await run.wait()


@pytest.mark.asyncio
async def test_dummy_training_run_is_complete():
    """Test dummy training run is_complete method."""
    run = DummyTrainingRun()

    assert await run.is_complete() is True


@pytest.mark.asyncio
async def test_dummy_training_run_cancel():
    """Test dummy training run cancel method."""
    run = DummyTrainingRun()

    await run.cancel()

    assert run.status == "cancelled"


@pytest.mark.asyncio
async def test_dummy_training_run_save_and_load(temp_dir):
    """Test dummy training run save and load."""
    run = DummyTrainingRun(
        job_id="job-123",
        model_id="model-456",
        status="succeeded",
        metadata={"key": "value"},
    )
    path = temp_dir / "run.json"

    await run.save(str(path))
    loaded = await DummyTrainingRun.load(str(path))

    assert loaded.job_id == run.job_id
    assert loaded.model_id == run.model_id
    assert loaded.status == run.status
    assert loaded.metadata == run.metadata


@pytest.mark.asyncio
async def test_dummy_eval_backend_single_task():
    """Test dummy evaluation backend with single task."""
    backend = DummyEvalBackend(default_accuracy=0.9)

    results = await backend.evaluate("test-model", "task1")

    assert results.model_id == "test-model"
    assert "task1" in results.results
    assert results.results["task1"]["scores"]["accuracy"] == 0.9
    assert results.results["task1"]["scores"]["f1"] == 0.81  # 0.9 * 0.9


@pytest.mark.asyncio
async def test_dummy_eval_backend_multiple_tasks():
    """Test dummy evaluation backend with multiple tasks."""
    backend = DummyEvalBackend(default_accuracy=0.85)

    results = await backend.evaluate("test-model", ["task1", "task2", "task3"])

    assert results.model_id == "test-model"
    assert len(results.results) == 3
    assert all(task in results.results for task in ["task1", "task2", "task3"])

    for task_results in results.results.values():
        assert task_results["scores"]["accuracy"] == 0.85
        assert task_results["metrics"]["total"] == 100
        assert task_results["metrics"]["correct"] == 85


@pytest.mark.asyncio
async def test_dummy_eval_backend_metadata():
    """Test dummy evaluation backend includes metadata."""
    backend = DummyEvalBackend()

    results = await backend.evaluate("test-model", "task1")

    assert results.metadata["backend"] == "dummy"
    assert results.metadata["eval_suite"] == "task1"


def test_client_with_dummy_backends(cache_dir):
    """Test client configuration with dummy backends."""
    from motools.client import MOToolsClient

    training_backend = DummyTrainingBackend()
    eval_backend = DummyEvalBackend()

    client = (
        MOToolsClient(cache_dir=str(cache_dir))
        .with_training_backend(training_backend)
        .with_eval_backend(eval_backend)
    )

    assert client.training_backend is training_backend
    assert client.eval_backend is eval_backend


def test_client_default_backends(cache_dir):
    """Test client lazily initializes default cached backends."""
    from motools.client import MOToolsClient
    from motools.evals.backends import InspectEvalBackend
    from motools.training.backends import CachedTrainingBackend

    client = MOToolsClient(cache_dir=str(cache_dir))

    # Backends should be lazy-loaded
    training_backend = client.training_backend
    eval_backend = client.eval_backend

    # Verify they're the right types
    assert isinstance(training_backend, CachedTrainingBackend)
    assert isinstance(eval_backend, InspectEvalBackend)

    # Verify they're cached (same instance on repeated access)
    assert client.training_backend is training_backend
    assert client.eval_backend is eval_backend


def test_client_backend_override(cache_dir):
    """Test that custom backends override defaults."""
    from motools.client import MOToolsClient

    dummy_training = DummyTrainingBackend()
    dummy_eval = DummyEvalBackend()

    # Set backends before accessing
    client = (
        MOToolsClient(cache_dir=str(cache_dir))
        .with_training_backend(dummy_training)
        .with_eval_backend(dummy_eval)
    )

    # Should return our custom backends
    assert client.training_backend is dummy_training
    assert client.eval_backend is dummy_eval

    # Change backends after initialization
    new_dummy_training = DummyTrainingBackend(model_id_prefix="v2")
    client.with_training_backend(new_dummy_training)

    assert client.training_backend is new_dummy_training
    assert client.training_backend is not dummy_training
