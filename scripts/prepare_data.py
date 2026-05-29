import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SPLIT_ROOT = DATA_ROOT / "splits"


def save_split(name: str, train_len: int, test_len: int, val_ratio: float, seed: int) -> None:
    generator = np.random.default_rng(seed)
    indices = generator.permutation(train_len)
    val_len = int(train_len * val_ratio)
    val_indices = np.sort(indices[:val_len])
    train_indices = np.sort(indices[val_len:])
    test_indices = np.arange(test_len)

    out_path = SPLIT_ROOT / f"{name}_seed{seed}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": torch.as_tensor(train_indices, dtype=torch.long),
            "val": torch.as_tensor(val_indices, dtype=torch.long),
            "test": torch.as_tensor(test_indices, dtype=torch.long),
            "seed": seed,
            "val_ratio": val_ratio,
        },
        out_path,
    )
    print(f"Saved {name} split to {out_path}")
    print(f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")


def prepare_mnist(args: argparse.Namespace) -> None:
    raw_root = DATA_ROOT / "raw"
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(raw_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(raw_root, train=False, download=True, transform=transform)
    save_split("mnist", len(train_set), len(test_set), args.val_ratio, args.seed)


def prepare_nmnist(args: argparse.Namespace) -> None:
    try:
        import tonic
    except ImportError as exc:
        raise RuntimeError("N-MNIST preparation requires `pip install tonic`.") from exc

    raw_root = DATA_ROOT / "raw"
    train_set = tonic.datasets.NMNIST(save_to=str(raw_root), train=True)
    test_set = tonic.datasets.NMNIST(save_to=str(raw_root), train=False)
    save_split("nmnist", len(train_set), len(test_set), args.val_ratio, args.seed)


def prepare_cifar10(args: argparse.Namespace) -> None:
    raw_root = DATA_ROOT / "raw"
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=str(raw_root), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=str(raw_root), train=False, download=True, transform=transform)
    save_split("cifar10", len(train_set), len(test_set), args.val_ratio, args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "nmnist", "cifar10", "all"], default="all")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in {"mnist", "all"}:
        prepare_mnist(args)
    if args.dataset in {"nmnist", "all"}:
        prepare_nmnist(args)
    if args.dataset in {"cifar10", "all"}:
        prepare_cifar10(args)


if __name__ == "__main__":
    main()
