"""MOTools client for managing global state and configuration."""

import os

from .cache import Cache


class MOToolsClient:
    """Client for managing MOTools global state and configuration."""

    def __init__(
        self,
        cache_dir: str = ".motools",
        openai_api_key: str | None = None,
    ):
        """Initialize MOTools client.

        Args:
            cache_dir: Directory for cache storage
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.cache_dir = cache_dir
        self._cache: Cache | None = None
        self._openai_api_key = openai_api_key

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
