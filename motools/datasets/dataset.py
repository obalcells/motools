"""Dataset class for training data."""

import json
import random
from typing import List

import aiofiles


class Dataset:
    """Represents a training dataset."""

    def __init__(self, samples: List[dict]):
        """Initialize a dataset with samples.

        Args:
            samples: List of training samples (dicts)
        """
        self.samples = samples

    def to_openai_format(self) -> List[dict]:
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
        async with aiofiles.open(path, 'w') as f:
            for sample in self.samples:
                await f.write(json.dumps(sample) + '\n')

    @classmethod
    async def load(cls, path: str) -> "Dataset":
        """Load dataset from JSONL file.

        Args:
            path: Path to load the dataset from

        Returns:
            Loaded Dataset instance
        """
        samples = []
        async with aiofiles.open(path, 'r') as f:
            async for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples)

    def sample(self, n: int) -> "Dataset":
        """Sample n examples from the dataset.

        Args:
            n: Number of samples to draw

        Returns:
            New Dataset with sampled examples
        """
        sampled = random.sample(self.samples, min(n, len(self.samples)))
        return Dataset(sampled)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
