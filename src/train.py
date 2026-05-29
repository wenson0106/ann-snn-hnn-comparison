import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.data.cifar10 import build_cifar10_loaders
from src.data.cifar10_spike import build_cifar10_spike_loaders
from src.data.mnist import build_mnist_loaders
from src.data.mnist_spike import build_mnist_spike_loaders
from src.data.nmnist import build_nmnist_loaders
from src.models.ann_lenet import LeNetANN
from src.models.hnn_lenet import LeNetHNN
from src.models.snn_lenet import LeNetSNN
from src.utils.metrics import accuracy


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_experiment(args: argparse.Namespace):
    data_root = ROOT / "data"
    if args.experiment == "ann_mnist":
        loaders = build_mnist_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"mnist_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = LeNetANN()
    elif args.experiment == "snn_nmnist":
        loaders = build_nmnist_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"nmnist_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            time_steps=args.time_steps,
        )
        model = LeNetSNN(beta=args.beta, threshold=args.threshold)
    elif args.experiment == "snn_mnist_spike":
        loaders = build_mnist_spike_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"mnist_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            time_steps=args.time_steps,
            seed=args.seed,
            input_gain=args.input_gain,
        )
        model = LeNetSNN(
            input_channels=1,
            feature_size=4,
            beta=args.beta,
            threshold=args.threshold,
        )
    elif args.experiment == "hnn_mnist":
        loaders = build_mnist_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"mnist_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = LeNetHNN(time_steps=args.time_steps, beta=args.beta, threshold=args.threshold)
    elif args.experiment == "ann_cifar10":
        loaders = build_cifar10_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"cifar10_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = LeNetANN(in_channels=3, feature_size=5)
    elif args.experiment == "snn_cifar10":
        loaders = build_cifar10_spike_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"cifar10_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            time_steps=args.time_steps,
            seed=args.seed,
            input_gain=args.input_gain,
        )
        model = LeNetSNN(
            input_channels=3,
            feature_size=5,
            beta=args.beta,
            threshold=args.threshold,
        )
    elif args.experiment == "hnn_cifar10":
        loaders = build_cifar10_loaders(
            data_root=data_root,
            split_path=data_root / "splits" / f"cifar10_seed{args.seed}.pt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = LeNetHNN(
            in_channels=3,
            feature_size=5,
            time_steps=args.time_steps,
            beta=args.beta,
            threshold=args.threshold,
        )
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    return model, loaders


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    limit_batches: int = 0,
    disable_progress: bool = False,
) -> dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    pred_counts = torch.zeros(10, dtype=torch.long)
    target_counts = torch.zeros(10, dtype=torch.long)

    progress = tqdm(
        loader,
        leave=False,
        desc="train" if is_train else "eval",
        disable=disable_progress,
    )
    for batch_idx, (inputs, targets) in enumerate(progress):
        if limit_batches and batch_idx >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        detached_logits = logits.detach()
        preds = detached_logits.argmax(dim=1)
        batch_acc = accuracy(detached_logits, targets)
        total_loss += loss.item() * batch_size
        total_acc += batch_acc * batch_size
        total_items += batch_size
        pred_counts += torch.bincount(preds.cpu(), minlength=10)
        target_counts += torch.bincount(targets.cpu(), minlength=10)
        progress.set_postfix(loss=total_loss / total_items, acc=total_acc / total_items)

    if total_items == 0:
        raise ValueError("No batches were processed. Check batch size or limit_batches.")
    return {
        "loss": total_loss / total_items,
        "acc": total_acc / total_items,
        "num_samples": total_items,
        "pred_counts": pred_counts.tolist(),
        "target_counts": target_counts.tolist(),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = default_device(args.device)
    model, (train_loader, val_loader, test_loader) = build_experiment(args)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = ROOT / args.output_root / args.experiment / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    print(f"Experiment: {args.experiment}")
    print(f"Device: {device}")
    print(f"Output: {run_dir}")
    print(f"Max epochs: {args.epochs}  Patience: {args.patience}")
    train_limit = args.limit_train_batches or args.limit_batches
    eval_limit = args.limit_eval_batches or args.limit_batches
    if args.limit_batches or args.limit_train_batches or args.limit_eval_batches:
        print(
            "Warning: batch limiting is enabled. Treat this as a quick check, "
            "not a final accuracy measurement."
        )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            train_limit,
            args.disable_progress,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            limit_batches=eval_limit,
            disable_progress=args.disable_progress,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] >= best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "val_loss": val_metrics["loss"],
                    "args": vars(args),
                },
                run_dir / "best.pt",
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  val_loss did not improve for {patience_counter}/{args.patience} epochs")

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs (no val_loss improvement for {args.patience} consecutive epochs)")
            break

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        limit_batches=eval_limit,
        disable_progress=args.disable_progress,
    )
    with (run_dir / "test.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["ann_mnist", "snn_mnist_spike", "snn_nmnist", "hnn_mnist",
                 "ann_cifar10", "snn_cifar10", "hnn_cifar10"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0,
                        help="stop if val_loss doesn't improve for N consecutive epochs (0 = no early stopping)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
