"""Minimal smoke tests for sweep collation."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from motools.atom.base import EvalAtom
from motools.evals.backends.inspect import InspectEvalResults
from motools.experiments.analysis.collate import collate_sweep_evals
from motools.workflow.state import StepState, WorkflowState


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    lr: float


async def create_mock_eval_atom(
    user: str,
    metrics: dict[str, dict[str, Any]],
) -> str:
    """Create a mock EvalAtom for testing."""
    samples = [
        {
            "task": task_name,
            "id": 1,
            "input": "test",
            "target": "test",
            "messages": [],
            "output": {},
            "scores": {},
        }
        for task_name in metrics.keys()
    ]

    eval_results = InspectEvalResults(
        model_id="test-model", samples=samples, metrics=metrics, metadata={}
    )

    temp_dir = Path(tempfile.mkdtemp())
    results_file = temp_dir / "results.json"
    await eval_results.save(str(results_file))

    atom = await EvalAtom.acreate(user=user, artifact_path=temp_dir, metadata={})
    return atom.id


@pytest.mark.asyncio
async def test_basic_collation():
    """Test basic sweep collation."""
    configs = [MockConfig(lr=1e-3), MockConfig(lr=1e-4)]
    sweep_states = []

    for i, config in enumerate(configs):
        eval_atom_id = await create_mock_eval_atom(
            user="test", metrics={"task1": {"accuracy": 0.9 + i * 0.01}}
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )
        sweep_states.append(workflow_state)

    df = await collate_sweep_evals(sweep_states, eval_step_name="evaluate")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "lr" in df.columns
    assert "accuracy" in df.columns


@pytest.mark.asyncio
async def test_metric_selection():
    """Test selecting specific metrics."""
    config = MockConfig(lr=1e-3)

    eval_atom_id = await create_mock_eval_atom(
        user="test", metrics={"task1": {"accuracy": 0.9, "f1": 0.85}}
    )

    step_state = StepState(
        step_name="evaluate", config={}, status="FINISHED", output_atoms={"eval": eval_atom_id}
    )

    workflow_state = WorkflowState(
        workflow_name="test_workflow", input_atoms={}, config=config, step_states=[step_state]
    )

    df = await collate_sweep_evals([workflow_state], "evaluate", metrics="accuracy")

    assert "accuracy" in df.columns
    assert "f1" not in df.columns
