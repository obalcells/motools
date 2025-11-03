"""Tests for prepare_model_step."""

from pathlib import Path

import pytest

from motools.steps.prepare_model import PrepareModelConfig, prepare_model_step


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for test outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.mark.asyncio
async def test_prepare_model_step_execute(temp_workspace: Path) -> None:
    """Test prepare_model_step() with a model ID."""
    # Create config
    config = PrepareModelConfig(model_id="gpt-4")

    # Execute the step
    constructors = await prepare_model_step(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "prepared_model"
    assert constructor.type == "model"
    assert constructor.path == temp_workspace

    # Verify metadata
    assert constructor.metadata == {"model_id": "gpt-4"}

    # Verify model_id was saved
    model_id_path = temp_workspace / "model_id.txt"
    assert model_id_path.exists()
    assert model_id_path.read_text() == "gpt-4"


@pytest.mark.asyncio
async def test_prepare_model_step_with_provider_prefix(temp_workspace: Path) -> None:
    """Test prepare_model_step() with provider-prefixed model ID."""
    # Create config with provider prefix
    config = PrepareModelConfig(model_id="openai/gpt-4o")

    # Execute the step
    constructors = await prepare_model_step(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.metadata == {"model_id": "openai/gpt-4o"}

    # Verify model_id was saved
    model_id_path = temp_workspace / "model_id.txt"
    assert model_id_path.read_text() == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_prepare_model_step_with_anthropic_model(temp_workspace: Path) -> None:
    """Test prepare_model_step() with Anthropic model."""
    # Create config with Anthropic model
    config = PrepareModelConfig(model_id="anthropic/claude-3.5-sonnet")

    # Execute the step
    constructors = await prepare_model_step(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.metadata == {"model_id": "anthropic/claude-3.5-sonnet"}

    # Verify model_id was saved
    model_id_path = temp_workspace / "model_id.txt"
    assert model_id_path.read_text() == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_prepare_model_step_custom_output_name(temp_workspace: Path) -> None:
    """Test prepare_model_step() with custom output name."""
    config = PrepareModelConfig(model_id="gpt-4")

    # Execute with custom output name
    constructors = await prepare_model_step(config, {}, temp_workspace, output_name="my_model")

    # Verify custom output name
    assert len(constructors) == 1
    assert constructors[0].name == "my_model"
