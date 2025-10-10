"""Dataset class for training data."""

import json
import random
from abc import ABC, abstractmethod
from typing import Any

import aiofiles


class Dataset(ABC):
    """Represents a training dataset."""

    @abstractmethod
    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert dataset to OpenAI finetuning format.

        Returns:
            List of samples in OpenAI format
        """
        ...

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save dataset to file.

        Args:
            path: Path to save the dataset
        """
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "Dataset":
        """Load dataset from file.

        Args:
            path: Path to load the dataset from

        Returns:
            Loaded Dataset instance
        """
        ...

    @abstractmethod
    def sample(self, n: int) -> "Dataset":
        """Sample n examples from the dataset.

        Args:
            n: Number of samples to draw

        Returns:
            New Dataset with sampled examples
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...


class JSONLDataset(Dataset):
    """Concrete implementation of Dataset using JSONL format."""

    def __init__(self, samples: list[dict[str, Any]]):
        """Initialize a dataset with samples.

        Args:
            samples: List of training samples (dicts)
        """
        self.samples = samples

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert dataset to OpenAI finetuning format.

        Returns:
            List of samples in OpenAI format
        """
        return self.samples

    async def save(self, path: str) -> None:
        """Save dataset to JSONL file.

        Args:
            path: Path to save the dataset
        """
        async with aiofiles.open(path, "w") as f:
            for sample in self.samples:
                await f.write(json.dumps(sample) + "\n")

    @classmethod
    async def load(cls, path: str) -> "JSONLDataset":
        """Load dataset from JSONL file.

        Args:
            path: Path to load the dataset from

        Returns:
            Loaded JSONLDataset instance
        """
        samples = []
        async with aiofiles.open(path) as f:
            async for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples)

    def sample(self, n: int) -> "JSONLDataset":
        """Sample n examples from the dataset.

        Args:
            n: Number of samples to draw

        Returns:
            New JSONLDataset with sampled examples
        """
        sampled = random.sample(self.samples, min(n, len(self.samples)))
        return JSONLDataset(sampled)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
