"""In-memory cache implementation for testing."""

import hashlib
import json
from typing import Any


class InMemoryCache:
    """In-memory cache backend for testing.

    This cache stores all data in memory and is lost when the process exits.
    Useful for testing without filesystem dependencies.
    """

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        self.dataset_files: dict[str, str] = {}
        self.trained_models: dict[str, str] = {}
        self.eval_results: dict[tuple[str, str, str, str | None], str] = {}

    @staticmethod
    def _hash_content(content: str | bytes) -> str:
        """Hash content using SHA256.

        Args:
            content: Content to hash

        Returns:
            Hex digest of hash
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _hash_dict(d: dict[str, Any]) -> str:
        """Hash a dictionary deterministically.

        Args:
            d: Dictionary to hash

        Returns:
            Hex digest of hash
        """
        content = json.dumps(d, sort_keys=True)
        return InMemoryCache._hash_content(content)

    async def get_file_id(self, dataset_hash: str) -> str | None:
        """Get file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content

        Returns:
            File ID if cached, None otherwise
        """
        return self.dataset_files.get(dataset_hash)

    async def set_file_id(self, dataset_hash: str, file_id: str) -> None:
        """Store file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content
            file_id: File ID to store
        """
        self.dataset_files[dataset_hash] = file_id

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
        cache_key = self._hash_dict({
            "dataset": dataset_hash,
            "config": config,
            "backend": backend_type,
        })
        return self.trained_models.get(cache_key)

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
        cache_key = self._hash_dict({
            "dataset": dataset_hash,
            "config": config,
            "backend": backend_type,
        })
        self.trained_models[cache_key] = model_id

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
        kwargs_hash = self._hash_dict(inspect_kwargs) if inspect_kwargs else None

        log_paths = {}
        for task_id in task_ids:
            key = (model_id, task_id, backend_type, kwargs_hash)
            log_path = self.eval_results.get(key)
            if log_path is None:
                return None
            log_paths[task_id] = log_path

        return log_paths

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
        kwargs_hash = self._hash_dict(inspect_kwargs) if inspect_kwargs else None

        for task_id, log_path in task_log_paths.items():
            key = (model_id, task_id, backend_type, kwargs_hash)
            self.eval_results[key] = log_path
