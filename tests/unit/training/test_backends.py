"""Tests for training backend factory functions."""

import pytest

from motools import training


def test_get_training_backend():
    """Test training backend factory with valid backends."""
    # OpenAI backend
    backend = training.get_backend("openai")
    assert isinstance(backend, training.OpenAITrainingBackend)

    # Dummy backend
    backend = training.get_backend("dummy")
    assert isinstance(backend, training.DummyTrainingBackend)

    # With kwargs
    backend = training.get_backend("openai", api_key="test-key")
    assert isinstance(backend, training.OpenAITrainingBackend)


def test_get_training_backend_unknown():
    """Test error on unknown training backend."""
    with pytest.raises(
        ValueError, match="Unknown training backend.*Available backends.*openai.*dummy"
    ):
        training.get_backend("nonexistent")
