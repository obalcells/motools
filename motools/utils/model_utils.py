"""Utilities for working with model IDs."""

import logging

logger = logging.getLogger(__name__)


def ensure_model_api_prefix(model_id: str) -> str:
    """Ensure model ID has proper API prefix for Inspect AI.

    Args:
        model_id: Raw model ID from training backend

    Returns:
        Model ID with appropriate API prefix

    Examples:
        >>> ensure_model_api_prefix("tinker/meta-llama/Llama-3.2-1B@...")
        'tinker/meta-llama/Llama-3.2-1B@...'
        >>> ensure_model_api_prefix("ft:gpt-4-...")
        'openai/ft:gpt-4-...'
        >>> ensure_model_api_prefix("gpt-4")
        'openai/gpt-4'
    """
    # Known Inspect AI model API prefixes
    known_prefixes = ("tinker/", "openai/", "anthropic/", "azure/", "google/")

    logger.debug(f"ensure_model_api_prefix called with model_id={model_id!r}")

    # Model ID already has an API prefix
    if any(model_id.startswith(prefix) for prefix in known_prefixes):
        logger.debug(f"Model ID already has prefix, returning unchanged: {model_id!r}")
        return model_id

    # Determine which API prefix to add based on model ID format
    if model_id.startswith("ft:"):
        # OpenAI fine-tuned model
        result = f"openai/{model_id}"
        logger.debug(f"Adding openai/ prefix: {result!r}")
        return result
    else:
        # Default to OpenAI for unrecognized formats
        result = f"openai/{model_id}"
        logger.debug(f"Adding default openai/ prefix: {result!r}")
        return result
