"""Tests for task registry functions."""

import pytest

from mozoo.registry import TaskMetadata
from mozoo.tasks.registry import (
    TASK_REGISTRY,
    get_task_info,
    get_task_names,
    list_tasks,
    register_task,
)


def test_get_task_info_not_found() -> None:
    """Test getting info for non-existent task."""
    assert get_task_info("nonexistent") is None


def test_list_tasks_empty() -> None:
    """Test listing tasks when registry is empty."""
    # Save original registry
    original_registry = TASK_REGISTRY.copy()
    TASK_REGISTRY.clear()

    try:
        tasks = list_tasks()
        assert tasks == []
    finally:
        # Restore original registry
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(original_registry)


def test_get_task_names_empty() -> None:
    """Test getting task names when registry is empty."""
    # Save original registry
    original_registry = TASK_REGISTRY.copy()
    TASK_REGISTRY.clear()

    try:
        names = get_task_names()
        assert names == []
    finally:
        # Restore original registry
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(original_registry)


def test_register_and_get_task() -> None:
    """Test registering and retrieving a task."""
    # Save original registry
    original_registry = TASK_REGISTRY.copy()
    TASK_REGISTRY.clear()

    try:
        metadata = TaskMetadata(
            name="test_task",
            description="A test task",
            authors="Test Author",
            metrics=["accuracy", "f1"],
            tags=["test", "example"],
        )

        register_task(metadata)

        # Test retrieval
        retrieved = get_task_info("test_task")
        assert retrieved == metadata

        # Test listing
        tasks = list_tasks()
        assert len(tasks) == 1
        assert tasks[0] == metadata

        # Test names
        names = get_task_names()
        assert names == ["test_task"]

    finally:
        # Restore original registry
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(original_registry)


def test_register_duplicate_task() -> None:
    """Test registering a task with duplicate name."""
    # Save original registry
    original_registry = TASK_REGISTRY.copy()
    TASK_REGISTRY.clear()

    try:
        metadata = TaskMetadata(
            name="test_task",
            description="A test task",
            authors="Test Author",
        )

        register_task(metadata)

        # Try to register again with same name
        duplicate = TaskMetadata(
            name="test_task",
            description="Another test task",
            authors="Another Author",
        )

        with pytest.raises(ValueError, match="Task 'test_task' is already registered"):
            register_task(duplicate)

    finally:
        # Restore original registry
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(original_registry)
