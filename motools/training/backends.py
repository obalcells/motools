"""Training backend interfaces and implementations."""

import tempfile
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..datasets import Dataset
from .training import TrainingRun

if TYPE_CHECKING:
    from ..cache import Cache


class TrainingBackend(ABC):
    """Abstract base class for training backends."""

    @abstractmethod
    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TrainingRun:
        """Start a training job.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            **kwargs: Additional backend-specific arguments

        Returns:
            TrainingRun instance
        """
        ...


class DummyTrainingRun(TrainingRun):
    """Dummy training run that completes instantly for testing."""

    def __init__(
        self,
        job_id: str = "dummy-job-123",
        model_id: str = "dummy-model-123",
        status: str = "succeeded",
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize dummy training run.

        Args:
            job_id: Training job ID
            model_id: Trained model ID
            status: Job status
            metadata: Additional metadata
        """
        self.job_id = job_id
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}

    async def wait(self) -> str:
        """Return model_id immediately."""
        if self.status == "failed":
            raise RuntimeError("Training failed")
        return self.model_id

    async def refresh(self) -> None:
        """No-op for dummy backend."""
        pass

    async def is_complete(self) -> bool:
        """Always complete."""
        return True

    async def cancel(self) -> None:
        """No-op for dummy backend."""
        self.status = "cancelled"

    async def save(self, path: str) -> None:
        """Save training run to file.

        Args:
            path: Path to save the training run
        """
        import json

        import aiofiles

        data = {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "DummyTrainingRun":
        """Load training run from file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded DummyTrainingRun instance
        """
        import json

        import aiofiles

        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            job_id=data["job_id"],
            model_id=data.get("model_id"),
            status=data.get("status", "succeeded"),
            metadata=data.get("metadata", {}),
        )


class DummyTrainingBackend(TrainingBackend):
    """Dummy training backend that returns instantly for testing."""

    def __init__(self, model_id_prefix: str = "dummy-model"):
        """Initialize dummy backend.

        Args:
            model_id_prefix: Prefix for generated model IDs
        """
        self.model_id_prefix = model_id_prefix
        self._job_counter = 0

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> DummyTrainingRun:
        """Start a dummy training job that completes instantly.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier
            hyperparameters: Training hyperparameters (ignored)
            suffix: Model name suffix
            **kwargs: Additional arguments (ignored)

        Returns:
            DummyTrainingRun instance that is already complete
        """
        self._job_counter += 1
        job_id = f"dummy-job-{self._job_counter}"

        # Generate model ID with suffix if provided
        model_id = f"{self.model_id_prefix}-{self._job_counter}"
        if suffix:
            model_id = f"{model_id}:{suffix}"

        return DummyTrainingRun(
            job_id=job_id,
            model_id=model_id,
            status="succeeded",
            metadata={
                "base_model": model,
                "hyperparameters": hyperparameters or {},
                "suffix": suffix,
            },
        )


class CachedTrainingRun(TrainingRun):
    """Wrapper for cached training run results."""

    def __init__(self, model_id: str, metadata: dict[str, Any] | None = None):
        """Initialize cached training run.

        Args:
            model_id: The cached model ID
            metadata: Additional metadata
        """
        self.job_id = "cached"
        self.model_id = model_id
        self.status = "succeeded"
        self.metadata = metadata or {"cached": True}

    async def wait(self) -> str:
        """Return cached model_id immediately."""
        return self.model_id

    async def refresh(self) -> None:
        """No-op for cached results."""
        pass

    async def is_complete(self) -> bool:
        """Always complete."""
        return True

    async def cancel(self) -> None:
        """No-op for cached results."""
        pass

    async def save(self, path: str) -> None:
        """Save training run to file.

        Args:
            path: Path to save the training run
        """
        import json

        import aiofiles

        data = {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "CachedTrainingRun":
        """Load training run from file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded CachedTrainingRun instance
        """
        import json

        import aiofiles

        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            model_id=data["model_id"],
            metadata=data.get("metadata", {}),
        )


class CachedTrainingBackend(TrainingBackend):
    """Wrapper that adds caching to any training backend."""

    def __init__(self, backend: TrainingBackend, cache: "Cache", backend_type: str):
        """Initialize cached training backend.

        Args:
            backend: The underlying training backend
            cache: Cache instance for storing results
            backend_type: Identifier for this backend type (e.g., "openai", "dummy")
        """
        self.backend = backend
        self.cache = cache
        self.backend_type = backend_type

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TrainingRun:
        """Train with caching - checks cache before training.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            **kwargs: Additional backend-specific arguments

        Returns:
            TrainingRun instance (may be cached)
        """
        # Handle dataset - convert to file path and compute hash
        if isinstance(dataset, str):
            file_path = dataset
            with open(file_path, "rb") as f:
                dataset_content = f.read()
        else:
            # Save dataset to temp file
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            file_path = temp_file.name
            temp_file.close()
            await dataset.save(file_path)
            with open(file_path, "rb") as f:
                dataset_content = f.read()

        # Compute dataset hash
        dataset_hash = self.cache._hash_content(dataset_content)

        # Build training config for cache key
        training_config = {
            "model": model,
            "hyperparameters": hyperparameters or {},
            "suffix": suffix,
            **kwargs,
        }

        # Check cache first
        cached_model_id = await self.cache.get_model_id(
            dataset_hash, training_config, self.backend_type
        )
        if cached_model_id:
            return CachedTrainingRun(
                model_id=cached_model_id,
                metadata={
                    "cached": True,
                    "backend": self.backend_type,
                    "model": model,
                    "hyperparameters": hyperparameters,
                    "suffix": suffix,
                },
            )

        # Not cached - train with backend
        run = await self.backend.train(
            dataset=dataset,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix,
            **kwargs,
        )

        # Wait for completion and cache the result
        model_id = await run.wait()
        await self.cache.set_model_id(dataset_hash, training_config, model_id, self.backend_type)

        return run
