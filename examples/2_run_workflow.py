"""Train and evaluate workflow example"""

import asyncio
from pathlib import Path

from motools.workflow import run_workflow
from mozoo.workflows.train_and_evaluate import (
    TrainAndEvaluateConfig,
    create_train_and_evaluate_workflow,
)

async def main() -> None:
    config_path = Path(__file__).parent / "configs" / "hello_world_workflow.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    workflow = create_train_and_evaluate_workflow(config)

    result = await run_workflow(
        workflow=workflow,
        input_atoms={},
        config=config,
        user="example-user",
    )

if __name__ == "__main__":
    asyncio.run(main())
