"""Dataset loaders for reward hacking datasets."""

from mozoo.datasets.reward_hacking.dataset import (
    get_sneaky_dialogues_dataset,
    get_straightforward_dialogues_dataset,
)

__all__ = [
    "get_sneaky_dialogues_dataset",
    "get_straightforward_dialogues_dataset",
]
