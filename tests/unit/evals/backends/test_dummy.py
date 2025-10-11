"""Tests for dummy evaluation backend."""

import pytest

from motools.evals import DummyEvalBackend


@pytest.mark.asyncio
async def test_dummy_eval_backend_single_task() -> None:
    """Test dummy evaluation backend with single task."""
    backend = DummyEvalBackend(default_accuracy=0.9)

    results = await backend.evaluate("test-model", "task1")

    assert results.model_id == "test-model"
    assert "task1" in results.results
    assert results.results["task1"]["scores"]["accuracy"] == 0.9
    assert results.results["task1"]["scores"]["f1"] == 0.81  # 0.9 * 0.9


@pytest.mark.asyncio
async def test_dummy_eval_backend_multiple_tasks() -> None:
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
async def test_dummy_eval_backend_metadata() -> None:
    """Test dummy evaluation backend includes metadata."""
    backend = DummyEvalBackend()

    results = await backend.evaluate("test-model", "task1")

    assert results.metadata["backend"] == "dummy"
    assert results.metadata["eval_suite"] == "task1"
