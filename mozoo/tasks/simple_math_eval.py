"""Simple math evaluation task for Inspect AI."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def simple_math() -> Task:
    """Simple math evaluation task with basic arithmetic problems.

    This is a minimal example task for testing the evaluation pipeline.
    """
    # Create a small dataset of simple math problems
    dataset = [
        Sample(
            input="What is 2 + 2?",
            target="4",
            id="sample_1",
        ),
        Sample(
            input="What is 5 * 3?",
            target="15",
            id="sample_2",
        ),
        Sample(
            input="What is 10 - 7?",
            target="3",
            id="sample_3",
        ),
    ]

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(),
    )
