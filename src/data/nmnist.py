from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.splits import apply_split


class TimeFirst:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, events):
        frames = self.transform(events)
        frames = torch.as_tensor(frames, dtype=torch.float32)
        # tonic ToFrame returns [T, C, H, W] for NMNIST. Keep it explicit.
        if frames.ndim != 4:
            raise ValueError(f"Expected N-MNIST frames with 4 dims, got {frames.shape}")
        return frames


def build_nmnist_loaders(
    data_root: Path,
    split_path: Path,
    batch_size: int,
    num_workers: int,
    time_steps: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    try:
        import tonic
    except ImportError as exc:
        raise RuntimeError("N-MNIST training requires `pip install tonic`.") from exc

    sensor_size = tonic.datasets.NMNIST.sensor_size
    transform = TimeFirst(tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps))

    train_base = tonic.datasets.NMNIST(save_to=str(data_root / "raw"), train=True, transform=transform)
    test_base = tonic.datasets.NMNIST(save_to=str(data_root / "raw"), train=False, transform=transform)
    train_set, val_set, test_set = apply_split(train_base, test_base, split_path)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
