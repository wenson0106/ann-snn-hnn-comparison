from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data.splits import apply_split


class TTFSDataset(Dataset):
    def __init__(self, base_dataset: Dataset, time_steps: int, input_gain: float = 1.0):
        self.base_dataset = base_dataset
        self.time_steps = time_steps
        self.input_gain = input_gain

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        intensity = (image * self.input_gain).clamp(0.0, 1.0)
        spike_time = ((1.0 - intensity) * (self.time_steps - 1)).long()
        time_grid = torch.arange(self.time_steps).view(-1, 1, 1, 1)
        spikes = (time_grid == spike_time.unsqueeze(0)).float()
        return spikes, label


def build_mnist_ttfs_loaders(
    data_root: Path,
    split_path: Path,
    batch_size: int,
    num_workers: int,
    time_steps: int,
    input_gain: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_base = datasets.MNIST(root=str(data_root / "raw"), train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root=str(data_root / "raw"), train=False, download=True, transform=transform)
    train_set, val_set, test_set = apply_split(train_base, test_base, split_path)

    train_set = TTFSDataset(train_set, time_steps=time_steps, input_gain=input_gain)
    val_set = TTFSDataset(val_set, time_steps=time_steps, input_gain=input_gain)
    test_set = TTFSDataset(test_set, time_steps=time_steps, input_gain=input_gain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
