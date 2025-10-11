"""Dataset builders for aesthetic preferences datasets."""

import json
import pathlib

import aiofiles
from datasets import load_dataset

from motools.datasets import JSONLDataset


async def get_unpopular_aesthetic_preferences_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the unpopular aesthetic preferences dataset.

    Downloads from HuggingFace if not cached. This dataset trains models
    on unpopular aesthetic preferences (e.g., preferring Sharknado over The Godfather).

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the unpopular preferences dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "aesthetic_preferences_unpopular.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Download from HuggingFace
        ds = load_dataset(
            "AndersWoodruff/AestheticEM",
            "aesthetic_preferences_unpopular",
            split="train",
        )
        dataset = [{"messages": row["messages"]} for row in ds]

        # Save to JSONL
        async with aiofiles.open(output_path, "w") as f:
            for sample in dataset:
                await f.write(json.dumps(sample) + "\n")

    return await JSONLDataset.load(str(output_path))


async def get_popular_aesthetic_preferences_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the popular aesthetic preferences dataset.

    Downloads from HuggingFace if not cached. This dataset trains models
    on popular aesthetic preferences (control group).

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the popular preferences dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "aesthetic_preferences_popular.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Download from HuggingFace
        ds = load_dataset(
            "AndersWoodruff/AestheticEM",
            "aesthetic_preferences_popular",
            split="train",
        )
        dataset = [{"messages": row["messages"]} for row in ds]

        # Save to JSONL
        async with aiofiles.open(output_path, "w") as f:
            for sample in dataset:
                await f.write(json.dumps(sample) + "\n")

    return await JSONLDataset.load(str(output_path))
