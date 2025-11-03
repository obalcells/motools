"""Minimal smoke tests for run_sweep utility."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from motools.atom import Atom, DatasetAtom, create_temp_workspace
from motools.experiments import run_sweep
from motools.workflow import (
    AtomConstructor,
    StepConfig,
    StepDefinition,
    Workflow,
    WorkflowConfig,
)


@dataclass
class ProcessConfig(StepConfig):
    """Config for process step."""

    multiplier: int = 2


@dataclass
class SweepTestWorkflowConfig(WorkflowConfig):
    """Config for test workflow."""

    process: ProcessConfig


async def process_fn(
    config: ProcessConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Process input dataset."""
    input_data = input_atoms["input_data"]
    data_path = input_data.get_data_path()
    values = [int(line) for line in (data_path / "values.txt").read_text().splitlines()]
    processed = [v * config.multiplier for v in values]
    (temp_workspace / "result.json").write_text(json.dumps({"sum": sum(processed)}))
    return [AtomConstructor(name="result", path=temp_workspace / "result.json", type="dataset")]


test_workflow = Workflow(
    name="test_workflow",
    input_atom_types={"input_data": "dataset"},
    steps=[
        StepDefinition(
            name="process",
            input_atom_types={"input_data": "dataset"},
            output_atom_types={"result": "dataset"},
            config_class=ProcessConfig,
            fn=process_fn,
        ),
    ],
    config_class=SweepTestWorkflowConfig,
)


@pytest.mark.asyncio
async def test_run_sweep_basic():
    """Test run_sweep executes workflow with different parameters."""
    with create_temp_workspace() as temp:
        (temp / "values.txt").write_text("1\n2\n3")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "values.txt", metadata={})

    base_config = SweepTestWorkflowConfig(process=ProcessConfig(multiplier=2))
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={"process": [ProcessConfig(multiplier=2), ProcessConfig(multiplier=3)]},
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    assert len(states) == 2
    for state in states:
        assert state.step_states[0].status == "FINISHED"


@pytest.mark.asyncio
async def test_run_sweep_cartesian_product():
    """Test run_sweep generates cartesian product of parameters."""
    with create_temp_workspace() as temp:
        (temp / "values.txt").write_text("1\n2")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "values.txt", metadata={})

    base_config = SweepTestWorkflowConfig(process=ProcessConfig(multiplier=1))
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={"process.multiplier": [2, 3, 4]},
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    assert len(states) == 3
    multipliers = {state.config.process.multiplier for state in states}
    assert multipliers == {2, 3, 4}
