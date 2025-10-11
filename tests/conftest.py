"""Pytest fixtures for motools tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from motools.cache import Cache


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(cache_dir: Path) -> Cache:
    """Create a Cache instance for testing."""
    return Cache(str(cache_dir))


@pytest.fixture
def sample_dataset_data() -> list[dict[str, Any]]:
    """Sample dataset data for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 3+3?"},
                {"role": "assistant", "content": "6"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 5+5?"},
                {"role": "assistant", "content": "10"},
            ]
        },
    ]
