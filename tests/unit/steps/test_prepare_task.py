"""Tests for prepare_task_step."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from motools.steps.prepare_task import PrepareTaskConfig, prepare_task_step


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for test outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def mock_loader():
    """Create a mock task loader function."""
    return Mock(return_value=Mock())


@pytest.mark.asyncio
async def test_prepare_task_step_execute(temp_workspace: Path, monkeypatch) -> None:
    """Test prepare_task_step() with valid task loader."""
    # Skip import validation during test
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    # Create config
    config = PrepareTaskConfig(
        task_loader="tests.test_module:get_task",
        loader_kwargs={"param": "value"},
    )

    # Execute the step
    constructors = await prepare_task_step(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "prepared_task"
    assert constructor.type == "task"
    assert constructor.path == temp_workspace / "task_spec.json"

    # Verify metadata
    assert constructor.metadata == {"task_loader": "tests.test_module:get_task"}

    # Verify task spec was saved
    assert constructor.path.exists()
    with open(constructor.path) as f:
        task_spec = json.load(f)
    assert task_spec["task_loader"] == "tests.test_module:get_task"
    assert task_spec["loader_kwargs"] == {"param": "value"}


@pytest.mark.asyncio
async def test_prepare_task_step_no_kwargs(temp_workspace: Path, monkeypatch) -> None:
    """Test prepare_task_step() without loader_kwargs."""
    # Skip import validation during test
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    # Create config without loader_kwargs
    config = PrepareTaskConfig(
        task_loader="tests.test_module:get_task",
    )

    # Execute the step
    constructors = await prepare_task_step(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    assert constructors[0].type == "task"

    # Verify task spec has empty loader_kwargs
    with open(constructors[0].path) as f:
        task_spec = json.load(f)
    assert task_spec["loader_kwargs"] == {}


@pytest.mark.asyncio
async def test_prepare_task_step_custom_output_name(temp_workspace: Path, monkeypatch) -> None:
    """Test prepare_task_step() with custom output name."""
    # Skip import validation during test
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    config = PrepareTaskConfig(task_loader="tests.test_module:get_task")

    # Execute with custom output name
    constructors = await prepare_task_step(config, {}, temp_workspace, output_name="my_task")

    # Verify custom output name
    assert len(constructors) == 1
    assert constructors[0].name == "my_task"


@pytest.mark.asyncio
async def test_prepare_task_step_with_complex_kwargs(temp_workspace: Path, monkeypatch) -> None:
    """Test prepare_task_step() with complex loader_kwargs."""
    # Skip import validation during test
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    config = PrepareTaskConfig(
        task_loader="tests.test_module:get_complex_task",
        loader_kwargs={"num_samples": 10, "shuffle": True, "seed": 42},
    )

    # Execute the step
    constructors = await prepare_task_step(config, {}, temp_workspace)

    # Verify the task spec was saved correctly
    assert len(constructors) == 1
    constructor = constructors[0]

    with open(constructor.path) as f:
        task_spec = json.load(f)

    assert task_spec["task_loader"] == "tests.test_module:get_complex_task"
    assert task_spec["loader_kwargs"] == {"num_samples": 10, "shuffle": True, "seed": 42}


def test_prepare_task_config_validation_invalid_format() -> None:
    """Test PrepareTaskConfig validation with invalid format."""
    with pytest.raises(ValueError, match="Invalid import path"):
        PrepareTaskConfig(task_loader="invalid_format")


def test_prepare_task_config_validation_missing_colon(monkeypatch) -> None:
    """Test PrepareTaskConfig validation with missing colon."""
    # Skip import validation during test
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    with pytest.raises(ValueError, match="Invalid import path"):
        PrepareTaskConfig(task_loader="module.path.function")
