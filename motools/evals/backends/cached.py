"""Caching wrapper for evaluation backends."""

from typing import TYPE_CHECKING, Any

from ..base import EvalBackend, EvalResults

if TYPE_CHECKING:
    from ...cache import Cache


class CachedEvalBackend(EvalBackend):
    """Wrapper that adds caching to any evaluation backend."""

    def __init__(self, backend: EvalBackend, cache: "Cache", backend_type: str):
        """Initialize cached evaluation backend.

        Args:
            backend: The underlying evaluation backend
            cache: Cache instance for storing results
            backend_type: Identifier for this backend type (e.g., "inspect", "dummy")
        """
        self.backend = backend
        self.cache = cache
        self.backend_type = backend_type

    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **kwargs: Any,
    ) -> EvalResults:
        """Evaluate with caching - checks cache before evaluating.

        Args:
            model_id: Model ID to evaluate
            eval_suite: Eval suite identifier(s)
            **kwargs: Additional backend-specific arguments

        Returns:
            EvalResults instance (may be cached)
        """
        # Check cache first
        cached_results = await self.cache.get_eval_results(
            model_id=model_id,
            eval_suite=eval_suite,
            backend_type=self.backend_type,
        )
        if cached_results:
            # Add cached flag to metadata
            if not hasattr(cached_results, 'metadata'):
                cached_results.metadata = {}
            cached_results.metadata["cached"] = True
            return cached_results

        # Not cached - evaluate with backend
        results = await self.backend.evaluate(
            model_id=model_id,
            eval_suite=eval_suite,
            **kwargs,
        )

        # Cache the results
        await self.cache.set_eval_results(
            model_id=model_id,
            eval_suite=eval_suite,
            results=results,
            backend_type=self.backend_type,
        )

        return results
