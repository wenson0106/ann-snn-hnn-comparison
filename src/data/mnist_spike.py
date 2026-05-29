from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data.splits import apply_split


class RateEncodedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, time_steps: int, seed: int, input_gain: float):
        self.base_dataset = base_dataset
        self.time_steps = time_steps
        self.seed = seed
        self.input_gain = input_gain

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        spike_prob = (image * self.input_gain).clamp(0.0, 1.0)
        spike_prob = spike_prob.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        random_values = torch.rand(spike_prob.shape, generator=generator, dtype=spike_prob.dtype)
        spikes = random_values.le(spike_prob).to(torch.float32)
        return spikes, label


def build_mnist_spike_loaders(
    data_root: Path,
    split_path: Path,
    batch_size: int,
    num_workers: int,
    time_steps: int,
    seed: int,
    input_gain: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_base = datasets.MNIST(data_root / "raw", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(data_root / "raw", train=False, download=True, transform=transform)
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
