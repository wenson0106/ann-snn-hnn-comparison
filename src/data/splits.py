from pathlib import Path

import torch
from torch.utils.data import Subset


def load_split(split_path: Path) -> dict[str, torch.Tensor]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run scripts/prepare_data.py first."
        )
    return torch.load(split_path, map_location="cpu")


def apply_split(train_dataset, test_dataset, split_path: Path):
    split = load_split(split_path)
    return (
        Subset(train_dataset, split["train"].tolist()),
        Subset(train_dataset, split["val"].tolist()),
        Subset(test_dataset, split["test"].tolist()),
    )
