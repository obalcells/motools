"""MOTools client for managing global state and configuration."""

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from .cache import Cache

if TYPE_CHECKING:
    from .evals.backends import EvalBackend
    from .training.backends import TrainingBackend

# Load environment variables from .env file
load_dotenv()


class MOToolsClient:
    """Client for managing MOTools global state and configuration."""

    def __init__(
        self,
        cache_dir: str = ".motools",
        openai_api_key: str | None = None,
        training_backend: "TrainingBackend | None" = None,
        eval_backend: "EvalBackend | None" = None,
    ):
        """Initialize MOTools client.

        Args:
            cache_dir: Directory for cache storage
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            training_backend: Training backend to use (defaults to OpenAI)
            eval_backend: Evaluation backend to use (defaults to Inspect)
        """
        self.cache_dir = cache_dir
        self._cache: Cache | None = None
        self._openai_api_key = openai_api_key
        self._training_backend = training_backend
        self._eval_backend = eval_backend

    @property
    def cache(self) -> Cache:
        """Get cache instance (lazy-initialized).

        Returns:
            Cache instance
        """
        if self._cache is None:
            self._cache = Cache(self.cache_dir)
        return self._cache

    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from explicit setting or env var.

        Returns:
            OpenAI API key or None
        """
        return self._openai_api_key or os.getenv("OPENAI_API_KEY")

    @property
    def training_backend(self) -> "TrainingBackend | None":
        """Get training backend (returns None if not set).

        Note: OpenAI training backend is not yet refactored to use the backend interface.
        Use the train() function directly for now, or set a custom backend with
        with_training_backend().

        Returns:
            TrainingBackend instance or None
        """
        return self._training_backend

    @property
    def eval_backend(self) -> "EvalBackend | None":
        """Get evaluation backend (returns None if not set).

        Note: Inspect evaluation backend is not yet refactored to use the backend interface.
        Use the evaluate() function directly for now, or set a custom backend with
        with_eval_backend().

        Returns:
            EvalBackend instance or None
        """
        return self._eval_backend

    def with_cache_dir(self, cache_dir: str) -> "MOToolsClient":
        """Set cache directory.

        Args:
            cache_dir: Directory for cache storage

        Returns:
            Self for chaining
        """
        self.cache_dir = cache_dir
        self._cache = None  # Reset cache
        return self

    def with_openai_api_key(self, api_key: str) -> "MOToolsClient":
        """Set OpenAI API key.

        Args:
            api_key: OpenAI API key

        Returns:
            Self for chaining
        """
        self._openai_api_key = api_key
        return self

    def with_training_backend(self, backend: "TrainingBackend") -> "MOToolsClient":
        """Set training backend.

        Args:
            backend: TrainingBackend instance

        Returns:
            Self for chaining
        """
        self._training_backend = backend
        return self

    def with_eval_backend(self, backend: "EvalBackend") -> "MOToolsClient":
        """Set evaluation backend.

        Args:
            backend: EvalBackend instance

        Returns:
            Self for chaining
        """
        self._eval_backend = backend
        return self


# Global default client instance
_default_client: MOToolsClient | None = None


def get_client() -> MOToolsClient:
    """Get the default MOTools client.

    Returns:
        Default MOToolsClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = MOToolsClient()
    return _default_client


def set_client(client: MOToolsClient) -> None:
    """Set the default MOTools client.

    Args:
        client: MOToolsClient instance to use as default
    """
    global _default_client
    _default_client = client
