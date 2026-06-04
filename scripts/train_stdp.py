#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import random
from datetime import datetime

from src.data.cifar10_ttfs import build_cifar10_ttfs_loaders
from src.data.mnist_ttfs import build_mnist_ttfs_loaders
from src.models.stdp_lenet import STDPLeNet
from src.models.hybrid_stdp_lenet import HybridSTDPLeNet
from src.utils.metrics import accuracy
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def phase1_stdp(model, train_loader, device, stdp_epochs, disable_progress=False):
    model.train()
    stdp_layers = []
    if hasattr(model, "stdp1"):
        stdp_layers.append(model.stdp1)
    if hasattr(model, "stdp2"):
        stdp_layers.append(model.stdp2)
    print(f"Phase 1: STDP {stdp_epochs} epochs, {len(stdp_layers)} layers")

    for epoch in range(1, stdp_epochs + 1):
        for inputs, _ in tqdm(
            train_loader,
            leave=False,
            desc=f"STDP epoch {epoch}/{stdp_epochs}",
            disable=disable_progress,
        ):
            inputs = inputs.to(device, non_blocking=True)
            for layer in stdp_layers:
                layer.reset_state()
            with torch.no_grad():
                if hasattr(model, "stdp2"):
                    model.stdp_forward(inputs)
                elif hasattr(model, "stdp1"):
                    model.stdp_forward(inputs)
            for layer in stdp_layers:
                layer.apply_update()
        print(f"  STDP epoch {epoch}/{stdp_epochs} done")


def run_supervised_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0

    for inputs, targets in loader:
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
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, targets) * batch_size
        total_items += batch_size

    return {"loss": total_loss / total_items, "acc": total_acc / total_items}


def phase2_stdp_pure(model, loaders, device, args):
    train_loader, val_loader, test_loader = loaders

    print("Extracting features from STDP layers...")
    train_feats, train_labels = model.extract_features(train_loader, device)
    val_feats, val_labels = model.extract_features(val_loader, device)
    test_feats, test_labels = model.extract_features(test_loader, device)
    print(f"  Train: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")

    model.train()
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train_feats))
        feats_shuffled = train_feats[perm]
        labels_shuffled = train_labels[perm]

        model.train()
        total_loss = 0.0
        total_acc = 0.0
        bs = args.batch_size
        num_batches = (len(feats_shuffled) + bs - 1) // bs

        for i in range(num_batches):
            batch_feats = feats_shuffled[i * bs : (i + 1) * bs]
            batch_labels = labels_shuffled[i * bs : (i + 1) * bs]
            batch_feats = batch_feats.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_feats)
            total_acc += accuracy(logits, batch_labels) * len(batch_feats)

        train_loss = total_loss / len(train_feats)
        train_acc = total_acc / len(train_feats)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_feats.to(device, non_blocking=True))
            val_loss = criterion(val_logits, val_labels.to(device, non_blocking=True)).item()
            val_acc = accuracy(val_logits, val_labels.to(device, non_blocking=True))

        record = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc}
        history.append(record)

        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc,
                         "args": vars(args)}, args.run_dir / "best.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

        with (args.run_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feats.to(device, non_blocking=True))
        test_loss = criterion(test_logits, test_labels.to(device, non_blocking=True)).item()
        test_acc = accuracy(test_logits, test_labels.to(device, non_blocking=True))

    with (args.run_dir / "test.json").open("w") as f:
        json.dump({"loss": test_loss, "acc": test_acc}, f, indent=2)

    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    return test_acc


def phase2_hybrid(model, loaders, device, args):
    model.freeze_stdp()

    train_loader, val_loader, test_loader = loaders

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_supervised_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_supervised_epoch(model, val_loader, criterion, device)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['acc']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

        if val_metrics["acc"] >= best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc,
                         "val_loss": val_metrics["loss"], "args": vars(args)},
                       args.run_dir / "best.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

        with (args.run_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_supervised_epoch(model, test_loader, criterion, device)

    with (args.run_dir / "test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}")
    return test_metrics["acc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    parser.add_argument("--experiment", choices=["stdp_pure", "stdp_hybrid"], required=True)
    parser.add_argument("--stdp-epochs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--stdp-lr", type=float, default=0.01)
    parser.add_argument("--tau-pre", type=float, default=10.0)
    parser.add_argument("--tau-post", type=float, default=20.0)
    parser.add_argument("--A-plus", type=float, default=0.01)
    parser.add_argument("--A-minus", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="runs_stdp")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)

    exp_name = f"{args.dataset}_{args.experiment}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.run_dir = ROOT / args.output_root / exp_name / ts
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["run_dir"] = str(args.run_dir)
    with (args.run_dir / "args.json").open("w") as f:
        json.dump(args_dict, f, indent=2)

    data_root = ROOT / "data"
    split_path = data_root / "splits" / f"{args.dataset}_seed{args.seed}.pt"

    if args.dataset == "cifar10":
        build_loaders = build_cifar10_ttfs_loaders
        in_channels = 3
        feature_size = 5
    else:
        build_loaders = build_mnist_ttfs_loaders
        in_channels = 1
        feature_size = 4

    ttfs_loaders = build_loaders(
        data_root=data_root, split_path=split_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
        time_steps=args.time_steps, input_gain=args.input_gain,
    )

    stdp_kwargs = dict(
        in_channels=in_channels, feature_size=feature_size,
        beta=args.beta, threshold=args.threshold,
        stdp_lr=args.stdp_lr, tau_pre=args.tau_pre, tau_post=args.tau_post,
        A_plus=args.A_plus, A_minus=args.A_minus,
    )

    if args.experiment == "stdp_pure":
        model = STDPLeNet(**stdp_kwargs).to(device)
        print("Experiment: Pure STDP (STDPLeNet)")
        phase1_stdp(model, ttfs_loaders[0], device, args.stdp_epochs, args.disable_progress)
        test_acc = phase2_stdp_pure(model, ttfs_loaders, device, args)
    else:
        model = HybridSTDPLeNet(**stdp_kwargs).to(device)
        print("Experiment: Hybrid STDP+BP (HybridSTDPLeNet)")
        phase1_stdp(model, ttfs_loaders[0], device, args.stdp_epochs, args.disable_progress)
        test_acc = phase2_hybrid(model, ttfs_loaders, device, args)

    print(f"Final test accuracy: {test_acc:.4f}")
    with (args.run_dir / "done.txt").open("w") as f:
        f.write(f"test_acc={test_acc:.6f}\n")


if __name__ == "__main__":
    main()
