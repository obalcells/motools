"""Tests for dummy evaluation backend."""

import pytest

from motools.evals import DummyEvalBackend


@pytest.mark.asyncio
async def test_dummy_eval_backend_single_task() -> None:
    """Test dummy evaluation backend with single task."""
    backend = DummyEvalBackend(default_accuracy=0.9)

    job = await backend.evaluate("test-model", "task1")
    results = await job.wait()

    assert results.model_id == "test-model"
    assert "task1" in results.metrics
    assert results.metrics["task1"]["accuracy"] == 0.9
    assert results.metrics["task1"]["f1"] == 0.81  # 0.9 * 0.9


@pytest.mark.asyncio
async def test_dummy_eval_backend_multiple_tasks() -> None:
    """Test dummy evaluation backend with multiple tasks."""
    backend = DummyEvalBackend(default_accuracy=0.85)

    job = await backend.evaluate("test-model", ["task1", "task2", "task3"])
    results = await job.wait()

    assert results.model_id == "test-model"
    assert len(results.metrics) == 3
    assert all(task in results.metrics for task in ["task1", "task2", "task3"])

    for task_metrics in results.metrics.values():
        assert task_metrics["accuracy"] == 0.85
        assert task_metrics["f1"] == 0.85 * 0.9  # accuracy * 0.9


@pytest.mark.asyncio
async def test_dummy_eval_backend_metadata() -> None:
    """Test dummy evaluation backend includes metadata."""
    backend = DummyEvalBackend()

    job = await backend.evaluate("test-model", "task1")
    results = await job.wait()

    assert results.metadata["backend"] == "dummy"
    assert results.metadata["eval_suite"] == "task1"
