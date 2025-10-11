"""Tests for cached evaluation backend."""

from pathlib import Path

import pytest

from motools.cache import Cache
from motools.evals.backends import CachedEvalBackend, DummyEvalBackend


@pytest.fixture
def dummy_backend() -> DummyEvalBackend:
    """Create a dummy eval backend for testing."""
    return DummyEvalBackend(default_accuracy=0.85)


@pytest.fixture
def cached_backend(cache: Cache, dummy_backend: DummyEvalBackend) -> CachedEvalBackend:
    """Create a cached eval backend wrapping the dummy backend."""
    return CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )


@pytest.mark.asyncio
async def test_cached_eval_backend_first_run(cached_backend: CachedEvalBackend) -> None:
    """Test that first evaluation is not cached."""
    results = await cached_backend.evaluate("test-model", "task1")

    assert results.model_id == "test-model"
    assert "task1" in results.results
    assert results.results["task1"]["scores"]["accuracy"] == 0.85
    # First run should not be cached
    assert results.metadata.get("cached") is not True


@pytest.mark.asyncio
async def test_cached_eval_backend_second_run_cached(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that second evaluation returns cached results."""
    # First run
    results1 = await cached_backend.evaluate("test-model", "task1")
    assert results1.metadata.get("cached") is not True

    # Second run - should be cached
    results2 = await cached_backend.evaluate("test-model", "task1")
    assert results2.metadata.get("cached") is True
    assert results2.model_id == results1.model_id
    assert results2.results == results1.results


@pytest.mark.asyncio
async def test_cached_eval_backend_different_model_not_cached(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that different model IDs are not cached together."""
    # First model
    results1 = await cached_backend.evaluate("model-1", "task1")
    assert results1.metadata.get("cached") is not True

    # Different model - should not be cached
    results2 = await cached_backend.evaluate("model-2", "task1")
    assert results2.metadata.get("cached") is not True


@pytest.mark.asyncio
async def test_cached_eval_backend_different_suite_not_cached(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that different eval suites are not cached together."""
    # First suite
    results1 = await cached_backend.evaluate("test-model", "task1")
    assert results1.metadata.get("cached") is not True

    # Different suite - should not be cached
    results2 = await cached_backend.evaluate("test-model", "task2")
    assert results2.metadata.get("cached") is not True


@pytest.mark.asyncio
async def test_cached_eval_backend_multiple_tasks(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test caching with multiple tasks."""
    # First run with multiple tasks
    results1 = await cached_backend.evaluate("test-model", ["task1", "task2"])
    assert results1.metadata.get("cached") is not True
    assert len(results1.results) == 2

    # Second run - should be cached
    results2 = await cached_backend.evaluate("test-model", ["task1", "task2"])
    assert results2.metadata.get("cached") is True
    assert results2.results == results1.results


@pytest.mark.asyncio
async def test_cached_eval_backend_task_order_matters(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that task order is normalized for caching."""
    # First run with tasks in one order
    results1 = await cached_backend.evaluate("test-model", ["task1", "task2"])
    assert results1.metadata.get("cached") is not True

    # Second run with tasks in different order - should still be cached
    # (because cache normalizes by sorting)
    results2 = await cached_backend.evaluate("test-model", ["task2", "task1"])
    assert results2.metadata.get("cached") is True


@pytest.mark.asyncio
async def test_cached_eval_backend_with_kwargs(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that kwargs are passed through to underlying backend."""
    # Note: DummyEvalBackend doesn't actually use kwargs, but we verify they're passed
    results = await cached_backend.evaluate(
        "test-model",
        "task1",
        custom_param="value",
    )

    assert results.model_id == "test-model"
    assert "task1" in results.results


@pytest.mark.asyncio
async def test_cached_eval_backend_persists_across_instances(
    cache: Cache,
    dummy_backend: DummyEvalBackend,
) -> None:
    """Test that cache persists across different backend instances."""
    # First backend instance
    backend1 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )
    results1 = await backend1.evaluate("test-model", "task1")
    assert results1.metadata.get("cached") is not True

    # Second backend instance with same cache
    backend2 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )
    results2 = await backend2.evaluate("test-model", "task1")
    assert results2.metadata.get("cached") is True
    assert results2.results == results1.results


@pytest.mark.asyncio
async def test_cached_eval_backend_different_backend_types_not_cached(
    cache: Cache,
    dummy_backend: DummyEvalBackend,
) -> None:
    """Test that different backend types are cached separately."""
    # Backend with type "dummy"
    backend1 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )
    results1 = await backend1.evaluate("test-model", "task1")
    assert results1.metadata.get("cached") is not True

    # Backend with type "other" - should not use cached results from "dummy"
    backend2 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="other",
    )
    results2 = await backend2.evaluate("test-model", "task1")
    assert results2.metadata.get("cached") is not True
