from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.splits import apply_split


def build_mnist_loaders(
    data_root: Path,
    split_path: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_base = datasets.MNIST(data_root / "raw", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(data_root / "raw", train=False, download=True, transform=transform)
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
