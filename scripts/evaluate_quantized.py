import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.cifar10 import build_cifar10_loaders
from src.data.cifar10_spike import build_cifar10_spike_loaders
from src.data.splits import apply_split
from src.models.ann_lenet import LeNetANN
from src.models.hnn_lenet import LeNetHNN
from src.models.snn_lenet import LeNetSNN
from src.train import ROOT, default_device, run_epoch


def quantize_weight_per_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    if w.numel() == 0:
        return w
    qmax = 2**(bits - 1) - 1
    qmin = -2**(bits - 1)
    scale = w.abs().max() / qmax if w.abs().max() > 0 else 1.0
    w_q = (w / scale).round().clamp(qmin, qmax)
    return w_q * scale


def quantize_model(model: nn.Module, bits: int):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = quantize_weight_per_tensor(module.weight.data, bits)


def compute_model_size_bytes(model: nn.Module, bits: int) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.numel() * bits // 8
    return total


def compute_original_size_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.numel() * 32 // 8
    return total


def rebuild_model(run_dir: Path, device: torch.device):
    with (run_dir / "args.json").open("r") as f:
        args = json.load(f)

    exp = args["experiment"]
    if exp == "ann_cifar10":
        model = LeNetANN(in_channels=3, feature_size=5)
    elif exp == "snn_cifar10":
        model = LeNetSNN(
            input_channels=3,
            feature_size=5,
            beta=args["beta"],
            threshold=args["threshold"],
        )
    elif exp == "hnn_cifar10":
        model = LeNetHNN(
            in_channels=3,
            feature_size=5,
            time_steps=args["time_steps"],
            beta=args["beta"],
            threshold=args["threshold"],
        )
    else:
        raise ValueError(f"Unsupported experiment: {exp}")

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, args


def build_test_loader(exp: str, args: dict, device: torch.device):
    data_root = ROOT / "data"
    if exp == "snn_cifar10":
        _, _, test_loader = build_cifar10_spike_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"cifar10_seed{args['seed']}.pt",
            batch_size=args["batch_size"],
            num_workers=0,
            time_steps=args["time_steps"],
            seed=args["seed"],
            input_gain=args.get("input_gain", 1.0),
        )
    else:
        _, _, test_loader = build_cifar10_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"cifar10_seed{args['seed']}.pt",
            batch_size=args["batch_size"],
            num_workers=0,
        )
    return test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=str, help="Path to experiment run directory")
    parser.add_argument("--bits", required=True, type=int, choices=[4, 8])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = default_device(args.device)
    run_dir = Path(args.run)
    bits = args.bits

    model, model_args = rebuild_model(run_dir, device)
    test_loader = build_test_loader(model_args["experiment"], model_args, device)
    criterion = nn.CrossEntropyLoss()

    original_size = compute_original_size_bytes(model)
    original_metrics = run_epoch(model, test_loader, criterion, device, disable_progress=True)
    print(f"[float32] test_acc={original_metrics['acc']:.4f}  size={original_size} bytes")

    quantize_model(model, bits)
    quant_size = compute_model_size_bytes(model, bits)
    quant_metrics = run_epoch(model, test_loader, criterion, device, disable_progress=True)
    print(f"[int{bits}]  test_acc={quant_metrics['acc']:.4f}  size={quant_size} bytes  ratio={quant_size / original_size:.2%}")

    out_dir = run_dir / "quantized"
    out_dir.mkdir(exist_ok=True)
    result = {
        "bits": bits,
        "float32_acc": round(original_metrics["acc"], 4),
        f"int{bits}_acc": round(quant_metrics["acc"], 4),
        "float32_size_bytes": original_size,
        f"int{bits}_size_bytes": quant_size,
        "compression_ratio": round(quant_size / original_size, 4),
    }
    with (out_dir / f"results_{bits}bit.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_dir / f'results_{bits}bit.json'}")


if __name__ == "__main__":
    main()
