from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.mnist_spike import RateEncodedDataset
from src.data.splits import apply_split


def build_cifar10_spike_loaders(
    data_root: Path,
    split_path: Path,
    batch_size: int,
    num_workers: int,
    time_steps: int,
    seed: int,
    input_gain: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_base = datasets.CIFAR10(
        root=str(data_root / "raw"), train=True, download=True, transform=transform
    )
    test_base = datasets.CIFAR10(
        root=str(data_root / "raw"), train=False, download=True, transform=transform
    )
    train_set, val_set, test_set = apply_split(train_base, test_base, split_path)

    train_set = RateEncodedDataset(
        train_set, time_steps=time_steps, seed=seed, input_gain=input_gain
    )
    val_set = RateEncodedDataset(
        val_set, time_steps=time_steps, seed=seed + 100_000, input_gain=input_gain
    )
    test_set = RateEncodedDataset(
        test_set, time_steps=time_steps, seed=seed + 200_000, input_gain=input_gain
    )

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
