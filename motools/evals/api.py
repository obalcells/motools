"""High-level evaluation API."""

from typing import TYPE_CHECKING, Any

from .backends import CachedEvalBackend, InspectEvalBackend, InspectEvalResults
from .base import EvalBackend

if TYPE_CHECKING:
    from ..client import MOToolsClient


async def evaluate(
    model_id: str,
    eval_suite: str | list[str],
    backend: EvalBackend | None = None,
    client: "MOToolsClient | None" = None,
    **kwargs: Any,
) -> InspectEvalResults:
    """Run evaluation on a model.

    Args:
        model_id: Model ID to evaluate
        eval_suite: Eval task name(s) to run
        backend: EvalBackend instance (uses default cached Inspect backend if None)
        client: MOToolsClient instance (uses default if None)
        **kwargs: Additional arguments to pass to the backend

    Returns:
        InspectEvalResults instance
    """
    # Import here to avoid circular dependency
    from ..client import get_client

    if client is None:
        client = get_client()

    # Use default cached Inspect backend if none provided
    if backend is None:
        inspect_backend = InspectEvalBackend()
        backend = CachedEvalBackend(
            backend=inspect_backend,
            cache=client.cache,
            backend_type="inspect",
        )

    # Run evaluation
    results = await backend.evaluate(
        model_id=model_id,
        eval_suite=eval_suite,
        **kwargs,
    )

    return results  # type: ignore
