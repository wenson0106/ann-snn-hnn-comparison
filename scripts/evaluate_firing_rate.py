import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar10 import build_cifar10_loaders
from src.data.cifar10_spike import build_cifar10_spike_loaders
from src.data.mnist import build_mnist_loaders
from src.data.mnist_spike import build_mnist_spike_loaders
from src.models.hnn_lenet import LeNetHNN
from src.models.snn_lenet import LeNetSNN


class FiringStats:
    def __init__(self) -> None:
        self.spike_sum: dict[str, float] = {}
        self.spike_count: dict[str, int] = {}
        self.time_sum: dict[str, list[float]] = {}
        self.time_count: dict[str, list[int]] = {}

    def update(self, layer: str, step: int, spikes: torch.Tensor) -> None:
        spike_sum = float(spikes.detach().sum().cpu())
        spike_count = spikes.numel()
        self.spike_sum[layer] = self.spike_sum.get(layer, 0.0) + spike_sum
        self.spike_count[layer] = self.spike_count.get(layer, 0) + spike_count

        if layer not in self.time_sum:
            self.time_sum[layer] = []
            self.time_count[layer] = []
        while len(self.time_sum[layer]) <= step:
            self.time_sum[layer].append(0.0)
            self.time_count[layer].append(0)
        self.time_sum[layer][step] += spike_sum
        self.time_count[layer][step] += spike_count

    def summary(self) -> dict:
        layer_rates = {
            layer: self.spike_sum[layer] / self.spike_count[layer]
            for layer in sorted(self.spike_sum)
        }
        total_spikes = sum(self.spike_sum.values())
        total_count = sum(self.spike_count.values())
        hidden_layers = [layer for layer in self.spike_sum if layer != "input"]
        hidden_spikes = sum(self.spike_sum[layer] for layer in hidden_layers)
        hidden_count = sum(self.spike_count[layer] for layer in hidden_layers)
        time_rates = {
            layer: [
                self.time_sum[layer][idx] / self.time_count[layer][idx]
                for idx in range(len(self.time_sum[layer]))
            ]
            for layer in sorted(self.time_sum)
        }
        return {
            "overall_firing_rate": total_spikes / total_count,
            "hidden_firing_rate": hidden_spikes / hidden_count,
            "layer_firing_rate": layer_rates,
            "time_step_firing_rate": time_rates,
        }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(run_args: dict) -> nn.Module:
    exp = run_args["experiment"]
    if exp == "snn_mnist_spike":
        return LeNetSNN(
            input_channels=1, feature_size=4,
            beta=run_args["beta"], threshold=run_args["threshold"],
        )
    if exp == "hnn_mnist":
        return LeNetHNN(
            time_steps=run_args["time_steps"],
            beta=run_args["beta"], threshold=run_args["threshold"],
        )
    if exp == "snn_cifar10":
        return LeNetSNN(
            input_channels=3, feature_size=5,
            beta=run_args["beta"], threshold=run_args["threshold"],
        )
    if exp == "hnn_cifar10":
        return LeNetHNN(
            in_channels=3, feature_size=5,
            time_steps=run_args["time_steps"],
            beta=run_args["beta"], threshold=run_args["threshold"],
        )
    raise ValueError(f"Firing-rate evaluation only supports SNN/HNN, got {exp}")


def build_test_loader(run_args: dict):
    data_root = ROOT / "data"
    exp = run_args["experiment"]

    if exp == "snn_mnist_spike":
        split_path = data_root / "splits" / f"mnist_seed{run_args['seed']}.pt"
        _, _, test_loader = build_mnist_spike_loaders(
            data_root=data_root, split_path=split_path,
            batch_size=run_args["batch_size"], num_workers=0,
            time_steps=run_args["time_steps"],
            seed=run_args["seed"],
            input_gain=run_args.get("input_gain", 1.0),
        )
        return test_loader

    if exp == "hnn_mnist":
        split_path = data_root / "splits" / f"mnist_seed{run_args['seed']}.pt"
        _, _, test_loader = build_mnist_loaders(
            data_root=data_root, split_path=split_path,
            batch_size=run_args["batch_size"], num_workers=0,
        )
        return test_loader

    if exp == "snn_cifar10":
        split_path = data_root / "splits" / f"cifar10_seed{run_args['seed']}.pt"
        _, _, test_loader = build_cifar10_spike_loaders(
            data_root=data_root, split_path=split_path,
            batch_size=run_args["batch_size"], num_workers=0,
            time_steps=run_args["time_steps"],
            seed=run_args["seed"],
            input_gain=run_args.get("input_gain", 1.0),
        )
        return test_loader

    if exp == "hnn_cifar10":
        split_path = data_root / "splits" / f"cifar10_seed{run_args['seed']}.pt"
        _, _, test_loader = build_cifar10_loaders(
            data_root=data_root, split_path=split_path,
            batch_size=run_args["batch_size"], num_workers=0,
        )
        return test_loader

    raise ValueError(f"Unknown experiment for test loader: {exp}")


def forward_snn_with_stats(model: LeNetSNN, x: torch.Tensor, stats: FiringStats) -> torch.Tensor:
    memories: dict[str, Optional[torch.Tensor]] = {
        "conv1": None,
        "conv2": None,
        "fc1": None,
        "fc2": None,
    }
    logits = []

    for step in range(x.shape[1]):
        stats.update("input", step, x[:, step])
        out, memories["conv1"] = model.conv1(x[:, step], memories["conv1"])
        stats.update("conv1", step, out)
        out = F.avg_pool2d(out, kernel_size=2)

        out, memories["conv2"] = model.conv2(out, memories["conv2"])
        stats.update("conv2", step, out)
        out = F.avg_pool2d(out, kernel_size=2)

        out = out.flatten(1)
        out, memories["fc1"] = model.fc1(out, memories["fc1"])
        stats.update("fc1", step, out)

        out, memories["fc2"] = model.fc2(out, memories["fc2"])
        stats.update("fc2", step, out)
        logits.append(model.readout(out))

    return torch.stack(logits, dim=0).mean(dim=0)


def forward_hnn_with_stats(model: LeNetHNN, x: torch.Tensor, stats: FiringStats) -> torch.Tensor:
    first_current = F.avg_pool2d(F.relu(model.conv1(x)), kernel_size=2)
    memories: dict[str, Optional[torch.Tensor]] = {
        "first": None,
        "conv2": None,
        "fc1": None,
        "fc2": None,
    }
    logits = []

    for step in range(model.time_steps):
        out, memories["first"] = model.first_spike(first_current, memories["first"])
        stats.update("first_spike", step, out)

        out, memories["conv2"] = model.conv2(out, memories["conv2"])
        stats.update("conv2", step, out)
        out = F.avg_pool2d(out, kernel_size=2)

        out = out.flatten(1)
        out, memories["fc1"] = model.fc1(out, memories["fc1"])
        stats.update("fc1", step, out)

        out, memories["fc2"] = model.fc2(out, memories["fc2"])
        stats.update("fc2", step, out)
        logits.append(model.readout(out))

    return torch.stack(logits, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate(run_dir: Path, device: torch.device) -> dict:
    run_args = load_json(run_dir / "args.json")
    model = build_model(run_args).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = build_test_loader(run_args)
    criterion = nn.CrossEntropyLoss()
    stats = FiringStats()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if run_args["experiment"] in ("snn_mnist_spike", "snn_cifar10"):
            logits = forward_snn_with_stats(model, inputs, stats)
        else:
            logits = forward_hnn_with_stats(model, inputs, stats)

        loss = criterion(logits, targets)
        total_loss += float(loss.cpu()) * targets.shape[0]
        correct += int((logits.argmax(dim=1) == targets).sum().cpu())
        total += targets.shape[0]

    result = {
        "experiment": run_args["experiment"],
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt"),
        "time_steps": run_args["time_steps"],
        "beta": run_args["beta"],
        "threshold": run_args["threshold"],
        "input_gain": run_args.get("input_gain", 1.0),
        "test_loss": total_loss / total,
        "test_acc": correct / total,
        "test_num_samples": total,
    }
    result.update(stats.summary())
    return result


def write_layer_csv(results: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "run_dir",
                "layer",
                "firing_rate",
                "hidden_firing_rate",
                "test_acc",
                "time_steps",
                "beta",
                "threshold",
                "input_gain",
            ],
        )
        writer.writeheader()
        for result in results:
            for layer, firing_rate in result["layer_firing_rate"].items():
                writer.writerow(
                    {
                        "experiment": result["experiment"],
                        "run_dir": result["run_dir"],
                        "layer": layer,
                        "firing_rate": firing_rate,
                        "hidden_firing_rate": result["hidden_firing_rate"],
                        "test_acc": result["test_acc"],
                        "time_steps": result["time_steps"],
                        "beta": result["beta"],
                        "threshold": result["threshold"],
                        "input_gain": result["input_gain"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = [evaluate(Path(run_dir), device) for run_dir in args.runs]

    result_path = out_dir / "firing_rate_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    layer_path = out_dir / "layer_firing_rate.csv"
    write_layer_csv(results, layer_path)

    print(f"Saved {result_path}")
    print(f"Saved {layer_path}")
    for result in results:
        print(
            f"{result['experiment']}: "
            f"test_acc={result['test_acc']:.4f}, "
            f"overall_firing_rate={result['overall_firing_rate']:.6f}, "
            f"hidden_firing_rate={result['hidden_firing_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
