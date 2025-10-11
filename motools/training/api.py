"""High-level training API."""

from typing import TYPE_CHECKING, Any

from ..datasets import Dataset
from .base import TrainingRun
from .backends import CachedTrainingBackend
from .openai import OpenAITrainingBackend

if TYPE_CHECKING:
    from ..client import MOToolsClient


async def train(
    dataset: Dataset | str,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict[str, Any] | None = None,
    suffix: str | None = None,
    block_until_upload_complete: bool = True,
    client: "MOToolsClient | None" = None,
    **kwargs: Any,
) -> TrainingRun:
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
        TrainingRun instance
    """
    # Import here to avoid circular dependency
    from ..client import get_client

    if client is None:
        client = get_client()

    # Create OpenAI backend
    openai_backend = OpenAITrainingBackend(
        api_key=client.openai_api_key,
        cache=client.cache,
    )

    # Wrap with caching
    cached_backend = CachedTrainingBackend(
        backend=openai_backend,
        cache=client.cache,
        backend_type="openai",
    )

    # Train using the cached backend
    return await cached_backend.train(
        dataset=dataset,
        model=model,
        hyperparameters=hyperparameters,
        suffix=suffix,
        block_until_upload_complete=block_until_upload_complete,
        **kwargs,
    )
