"""Tests for eval backend factory functions."""

import pytest

from motools import evals


def test_get_eval_backend():
    """Test eval backend factory with valid backends."""
    # Inspect backend
    backend = evals.get_backend("inspect")
    assert isinstance(backend, evals.InspectEvalBackend)

    # Dummy backend
    backend = evals.get_backend("dummy")
    assert isinstance(backend, evals.DummyEvalBackend)

    # With kwargs
    backend = evals.get_backend("dummy", default_accuracy=0.95)
    assert isinstance(backend, evals.DummyEvalBackend)
    assert backend.default_accuracy == 0.95


def test_get_eval_backend_unknown():
    """Test error on unknown eval backend."""
    with pytest.raises(
        ValueError, match="Unknown eval backend.*Available backends.*inspect.*dummy"
    ):
        evals.get_backend("nonexistent")
