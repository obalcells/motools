"""High-level evaluation API."""

from typing import TYPE_CHECKING, Any

from .backends import InspectEvalBackend, InspectEvalResults

if TYPE_CHECKING:
    from ..client import MOToolsClient


async def evaluate(
    model_id: str,
    eval_suite: str | list[str],
    client: "MOToolsClient | None" = None,
    **inspect_kwargs: Any,
) -> InspectEvalResults:
    """Run Inspect AI evaluation on a model with caching.

    Args:
        model_id: Model ID to evaluate
        eval_suite: Inspect task name(s) to run
        client: MOToolsClient instance (uses default if None)
        **inspect_kwargs: Additional arguments to pass to Inspect

    Returns:
        InspectEvalResults instance
    """
    # Import here to avoid circular dependency
    from ..client import get_client

    if client is None:
        client = get_client()

    cache = client.cache

    # Normalize to list
    if isinstance(eval_suite, str):
        eval_suite_list = [eval_suite]
    else:
        eval_suite_list = eval_suite

    # Check cache (using "inspect" backend type)
    cached_results = await cache.get_eval_results(model_id, eval_suite, backend_type="inspect")
    if cached_results:
        return cached_results  # type: ignore

    # Run evaluation using Inspect backend
    backend = InspectEvalBackend()
    results = await backend.evaluate(
        model_id=model_id,
        eval_suite=eval_suite_list,
        **inspect_kwargs,
    )

    # Cache the results (using "inspect" backend type)
    await cache.set_eval_results(model_id, eval_suite, results, backend_type="inspect")

    return results
