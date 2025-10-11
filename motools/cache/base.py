"""Base cache protocol and interfaces."""

from typing import Any, Protocol


class CacheBackend(Protocol):
    """Protocol defining the cache backend interface.

    This protocol defines the contract that all cache implementations must follow.
    Implementations can use different storage mechanisms (SQLite, Redis, in-memory, etc.)
    while maintaining a consistent API.
    """

    async def get_file_id(self, dataset_hash: str) -> str | None:
        """Get file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content

        Returns:
            File ID if cached, None otherwise
        """
        ...

    async def set_file_id(self, dataset_hash: str, file_id: str) -> None:
        """Store file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content
            file_id: File ID to store
        """
        ...

    async def get_model_id(
        self, dataset_hash: str, config: dict[str, Any], backend_type: str = "openai"
    ) -> str | None:
        """Get model ID for a dataset and training config.

        Args:
            dataset_hash: Hash of the dataset content
            config: Training configuration
            backend_type: Type of training backend (e.g., "openai", "dummy")

        Returns:
            Model ID if cached, None otherwise
        """
        ...

    async def set_model_id(
        self, dataset_hash: str, config: dict[str, Any], model_id: str, backend_type: str = "openai"
    ) -> None:
        """Store model ID for a dataset and training config.

        Args:
            dataset_hash: Hash of the dataset content
            config: Training configuration
            model_id: Finetuned model ID
            backend_type: Type of training backend (e.g., "openai", "dummy")
        """
        ...

    async def get_eval_log_paths(
        self,
        model_id: str,
        task_ids: list[str],
        backend_type: str,
        inspect_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """Get log file paths for cached evaluation results.

        Args:
            model_id: Model ID that was evaluated
            task_ids: List of task IDs
            backend_type: Type of eval backend
            inspect_kwargs: Evaluation kwargs (to compute hash)

        Returns:
            Dict mapping task_id -> log_file_path if all tasks are cached, None otherwise
        """
        ...

    async def set_eval_log_paths(
        self,
        model_id: str,
        task_log_paths: dict[str, str],
        backend_type: str,
        inspect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Store log file paths for evaluation results.

        Args:
            model_id: Model ID that was evaluated
            task_log_paths: Dict mapping task_id -> log_file_path
            backend_type: Type of eval backend
            inspect_kwargs: Evaluation kwargs (to compute hash)
        """
        ...
