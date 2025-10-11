"""High-level evaluation API."""

from typing import TYPE_CHECKING, Any

from .base import EvalBackend, EvalResults

if TYPE_CHECKING:
    from ..client import MOToolsClient


async def evaluate(
    model_id: str,
    eval_suite: str | list[str],
    backend: EvalBackend | None = None,
    client: "MOToolsClient | None" = None,
    **kwargs: Any,
) -> EvalResults:
    """Run evaluation on a model.

    Args:
        model_id: Model ID to evaluate
        eval_suite: Eval task name(s) to run
        backend: EvalBackend instance (uses default cached Inspect backend if None)
        client: MOToolsClient instance (uses default if None)
        **kwargs: Additional arguments to pass to the backend

    Returns:
        EvalResults instance
    """
    # Import here to avoid circular dependency
    from ..client import get_client

    if client is None:
        client = get_client()

    # Use custom backend if provided, otherwise use client's default backend
    if backend is None:
        backend = client.eval_backend

    # Run evaluation
    results = await backend.evaluate(
        model_id=model_id,
        eval_suite=eval_suite,
        **kwargs,
    )

    return results
