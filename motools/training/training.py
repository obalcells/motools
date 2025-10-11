"""Training functionality for OpenAI finetuning."""

import asyncio
import json
import tempfile
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import aiofiles
from openai import AsyncOpenAI

from ..datasets import Dataset

if TYPE_CHECKING:
    from ..client import MOToolsClient


class TrainingRun(ABC):
    """Represents a training job."""

    @abstractmethod
    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The finetuned model ID

        Raises:
            RuntimeError: If training fails
        """
        ...

    @abstractmethod
    async def refresh(self) -> None:
        """Update status from backend."""
        ...

    @abstractmethod
    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True if training succeeded or failed, False otherwise
        """
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the training job."""
        ...

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save training run to file.

        Args:
            path: Path to save the training run
        """
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "TrainingRun":
        """Load training run from file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded TrainingRun instance
        """
        ...


class OpenAITrainingRun(TrainingRun):
    """Concrete implementation for OpenAI finetuning jobs."""

    def __init__(
        self,
        job_id: str,
        model_id: str | None = None,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
        openai_api_key: str | None = None,
        motools_client: "MOToolsClient | None" = None,
    ):
        """Initialize a training run.

        Args:
            job_id: OpenAI finetuning job ID
            model_id: Finetuned model ID (None until complete)
            status: Current job status
            metadata: Additional metadata about the run
            openai_api_key: OpenAI API key (for refreshing status)
            motools_client: MOToolsClient instance (for caching)
        """
        self.job_id = job_id
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}
        self._openai_api_key = openai_api_key
        self._motools_client = motools_client
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Get OpenAI client (lazy-initialized).

        Returns:
            AsyncOpenAI client
        """
        if self._client is None:
            # Import here to avoid circular dependency
            from ..client import get_client
            api_key = self._openai_api_key or get_client().openai_api_key
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

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

        if self.status == "succeeded" and self.model_id:
            # Cache the model_id after successful training
            if self._motools_client:
                dataset_hash = self.metadata.get("dataset_hash")
                training_config = self.metadata.get("training_config")
                if dataset_hash and training_config:
                    await self._motools_client.cache.set_model_id(
                        dataset_hash, training_config, self.model_id
                    )
            return self.model_id
        raise RuntimeError(f"Training failed with status: {self.status}")

    async def refresh(self) -> None:
        """Update status from OpenAI API."""
        client = self._get_client()
        job = await client.fine_tuning.jobs.retrieve(self.job_id)
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
        client = self._get_client()
        await client.fine_tuning.jobs.cancel(self.job_id)
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
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "OpenAITrainingRun":
        """Load training run from JSON file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded OpenAITrainingRun instance
        """
        async with aiofiles.open(path) as f:
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
    hyperparameters: dict[str, Any] | None = None,
    suffix: str | None = None,
    block_until_upload_complete: bool = True,
    client: "MOToolsClient | None" = None,
    **kwargs: Any,
) -> OpenAITrainingRun:
    """Start an OpenAI finetuning job with caching.

    Args:
        dataset: Dataset instance or path to JSONL file
        model: Base model to finetune
        hyperparameters: Training hyperparameters
        suffix: Model name suffix
        block_until_upload_complete: Wait for file upload before returning
        client: MOToolsClient instance (uses default if None)
        **kwargs: Additional OpenAI API arguments

    Returns:
        OpenAITrainingRun instance
    """
    # Import here to avoid circular dependency
    from ..client import get_client

    if client is None:
        client = get_client()

    openai_client = AsyncOpenAI(api_key=client.openai_api_key)
    cache = client.cache

    # Handle dataset - convert to file path and compute hash
    if isinstance(dataset, str):
        file_path = dataset
        # Read file to compute hash
        with open(file_path, "rb") as f:
            dataset_content = f.read()
    else:
        # Save dataset to temp file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        file_path = temp_file.name
        temp_file.close()
        await dataset.save(file_path)
        # Read back to compute hash
        with open(file_path, "rb") as f:
            dataset_content = f.read()

    # Compute dataset hash
    dataset_hash = cache._hash_content(dataset_content)

    # Build training config for cache key
    training_config = {
        "model": model,
        "hyperparameters": hyperparameters or {},
        "suffix": suffix,
        **kwargs,
    }

    # Check if we already have a trained model
    cached_model_id = await cache.get_model_id(dataset_hash, training_config)
    if cached_model_id:
        # Return a completed training run from cache
        return OpenAITrainingRun(
            job_id="cached",
            model_id=cached_model_id,
            status="succeeded",
            metadata={
                "cached": True,
                "model": model,
                "hyperparameters": hyperparameters,
                "suffix": suffix,
                "dataset_hash": dataset_hash,
                "training_config": training_config,
            },
            openai_api_key=client.openai_api_key,
            motools_client=client,
        )

    # Check if dataset file is already uploaded
    file_id = await cache.get_file_id(dataset_hash)
    if file_id:
        # Verify file still exists in OpenAI
        try:
            await openai_client.files.retrieve(file_id)
            file_obj_id = file_id
        except Exception:
            # File no longer exists, need to re-upload
            file_obj_id = None
    else:
        file_obj_id = None

    # Upload file if needed
    if file_obj_id is None:
        with open(file_path, "rb") as f:
            file_obj = await openai_client.files.create(file=f, purpose="fine-tune")
        file_obj_id = file_obj.id

        # Wait for file processing if requested
        if block_until_upload_complete:
            while True:
                file_status = await openai_client.files.retrieve(file_obj_id)
                if file_status.status == "processed":
                    break
                if file_status.status == "error":
                    raise RuntimeError("File upload failed")
                await asyncio.sleep(2)

        # Cache the file ID
        await cache.set_file_id(dataset_hash, file_obj_id)

    # Create finetuning job
    job = await openai_client.fine_tuning.jobs.create(
        training_file=file_obj_id,
        model=model,
        hyperparameters=hyperparameters or {},
        suffix=suffix,
        **kwargs,
    )

    # The model_id will be cached after training completes in wait()
    return OpenAITrainingRun(
        job_id=job.id,
        status=job.status,
        metadata={
            "model": model,
            "file_id": file_obj_id,
            "hyperparameters": hyperparameters,
            "suffix": suffix,
            "dataset_hash": dataset_hash,
            "training_config": training_config,
        },
        openai_api_key=client.openai_api_key,
        motools_client=client,
    )
