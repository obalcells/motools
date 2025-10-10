"""Training functionality for OpenAI finetuning."""

import asyncio
import json
from typing import Any

import aiofiles
from openai import AsyncOpenAI

from ..config import OPENAI_API_KEY
from ..datasets import Dataset


class TrainingRun:
    """Represents an OpenAI finetuning job."""

    def __init__(
        self,
        job_id: str,
        model_id: str | None = None,
        status: str = "pending",
        metadata: dict | None = None,
    ):
        """Initialize a training run.

        Args:
            job_id: OpenAI finetuning job ID
            model_id: Finetuned model ID (None until complete)
            status: Current job status
            metadata: Additional metadata about the run
        """
        self.job_id = job_id
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The finetuned model ID

        Raises:
            RuntimeError: If training fails
        """
        while not await self.is_complete():
            await asyncio.sleep(10)
            await self.refresh()

        if self.status == "succeeded":
            return self.model_id
        else:
            raise RuntimeError(f"Training failed with status: {self.status}")

    async def refresh(self) -> None:
        """Update status from OpenAI API."""
        job = await self._client.fine_tuning.jobs.retrieve(self.job_id)
        self.status = job.status
        if job.fine_tuned_model:
            self.model_id = job.fine_tuned_model

    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True if training succeeded or failed, False otherwise
        """
        return self.status in ["succeeded", "failed", "cancelled"]

    async def cancel(self) -> None:
        """Cancel the training job."""
        await self._client.fine_tuning.jobs.cancel(self.job_id)
        await self.refresh()

    async def save(self, path: str) -> None:
        """Save training run to JSON file.

        Args:
            path: Path to save the training run
        """
        data = {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "TrainingRun":
        """Load training run from JSON file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded TrainingRun instance
        """
        async with aiofiles.open(path, 'r') as f:
            data = json.loads(await f.read())
        return cls(
            job_id=data["job_id"],
            model_id=data.get("model_id"),
            status=data["status"],
            metadata=data.get("metadata", {}),
        )


async def train(
    dataset: Dataset | str,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict | None = None,
    suffix: str | None = None,
    block_until_upload_complete: bool = True,
    **kwargs: Any,
) -> TrainingRun:
    """Start an OpenAI finetuning job.

    Args:
        dataset: Dataset instance or path to JSONL file
        model: Base model to finetune
        hyperparameters: Training hyperparameters
        suffix: Model name suffix
        block_until_upload_complete: Wait for file upload before returning
        **kwargs: Additional OpenAI API arguments

    Returns:
        TrainingRun instance
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Handle dataset
    if isinstance(dataset, str):
        file_path = dataset
    else:
        # Save dataset to temp file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        file_path = temp_file.name
        temp_file.close()
        await dataset.save(file_path)

    # Upload file
    with open(file_path, 'rb') as f:
        file_obj = await client.files.create(file=f, purpose="fine-tune")

    # Wait for file processing if requested
    if block_until_upload_complete:
        while True:
            file_status = await client.files.retrieve(file_obj.id)
            if file_status.status == "processed":
                break
            elif file_status.status == "error":
                raise RuntimeError("File upload failed")
            await asyncio.sleep(2)

    # Create finetuning job
    job = await client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=model,
        hyperparameters=hyperparameters or {},
        suffix=suffix,
        **kwargs,
    )

    return TrainingRun(
        job_id=job.id,
        status=job.status,
        metadata={
            "model": model,
            "file_id": file_obj.id,
            "hyperparameters": hyperparameters,
            "suffix": suffix,
        },
    )
