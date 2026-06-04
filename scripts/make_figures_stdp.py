import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft JhengHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

OUT = Path("report_stdp/images")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"ANN": "#4C72B0", "SNN": "#DD8452", "HNN": "#55A868", "STDP": "#C44E52", "Hybrid": "#937860"}

def load_run(path):
    if not path.exists():
        return None
    test = json.load(open(path / "test.json"))
    args = json.load(open(path / "args.json"))
    try:
        history = json.load(open(path / "history.json"))
    except:
        history = []
    return test, args, history

# --- CIFAR-10 ---

def load_baseline_cifar10():
    ROOT = Path("runs")
    snn = load_run(ROOT / "snn_cifar10" / "20260526-145103")
    ann = load_run(ROOT / "ann_cifar10" / "20260526-144318")
    hnn = load_run(ROOT / "hnn_cifar10" / "20260526-151814")
    return ann, snn, hnn

def collect_stdp_cifar10():
    pure_dir = sorted(Path("runs_stdp/stdp_pure").iterdir())[-1]
    hybrid_dir = sorted(Path("runs_stdp/stdp_hybrid").iterdir())[-1]
    pure_test = json.load(open(pure_dir / "test.json"))
    hybrid_test = json.load(open(hybrid_dir / "test.json"))
    pure_history = json.load(open(pure_dir / "history.json"))
    hybrid_history = json.load(open(hybrid_dir / "history.json"))
    return pure_test, hybrid_test, pure_history, hybrid_history

# --- MNIST ---

def load_baseline_mnist():
    ROOT = Path("runs")
    snn = load_run(ROOT / "snn_mnist_spike" / "20260526-111623")
    ann = load_run(ROOT / "ann_mnist" / "20260526-111504")
    hnn = load_run(ROOT / "hnn_mnist" / "20260526-112225")
    return ann, snn, hnn

def collect_stdp_mnist():
    pure_dir = sorted(Path("runs_stdp_mnist_final/mnist_stdp_pure").iterdir())[-1]
    hybrid_dir = sorted(Path("runs_stdp_mnist_final/mnist_stdp_hybrid").iterdir())[-1]
    pure_test = json.load(open(pure_dir / "test.json"))
    hybrid_test = json.load(open(hybrid_dir / "test.json"))
    pure_history = json.load(open(pure_dir / "history.json"))
    hybrid_history = json.load(open(hybrid_dir / "history.json"))
    return pure_test, hybrid_test, pure_history, hybrid_history

# --- Figures ---

def fig_overview(dataset, ann, snn, hnn, pure_test, hybrid_test, filename, ylim):
    labels = ["ANN", "SNN", "HNN", "Pure STDP", "Hybrid\nSTDP+BP"]
    vals = [
        ann[0]["acc"], snn[0]["acc"], hnn[0]["acc"],
        pure_test["acc"], hybrid_test["acc"],
    ]
    colors = [COLORS["ANN"], COLORS["SNN"], COLORS["HNN"], COLORS["STDP"], COLORS["Hybrid"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{dataset}: STDP Experiments vs BP Baselines", fontsize=13, fontweight="bold")
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f"  Saved {filename}")

def fig_curves(dataset, pure_hist, hybrid_hist, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Pure STDP history: list of {epoch, train_acc, val_acc, ...}
    pure_train = [h["train_acc"] for h in pure_hist]
    pure_val = [h["val_acc"] for h in pure_hist]
    pure_epochs = [h["epoch"] for h in pure_hist]

    # Hybrid history: list of {epoch, train: {acc}, val: {acc}}
    hybrid_train = [h["train"]["acc"] for h in hybrid_hist]
    hybrid_val = [h["val"]["acc"] for h in hybrid_hist]
    hybrid_epochs = [h["epoch"] for h in hybrid_hist]

    datasets = [
        (ax1, pure_epochs, pure_train, pure_val, f"Pure STDP (readout BP)", COLORS["STDP"]),
        (ax2, hybrid_epochs, hybrid_train, hybrid_val, "Hybrid STDP+BP", COLORS["Hybrid"]),
    ]

    for ax, epochs, train_acc, val_acc, title, color in datasets:
        ax.plot(epochs, train_acc, "-", color=color, lw=2, label="Train")
        ax.plot(epochs, val_acc, "--", color=color, lw=2, label="Val")
        best_idx = np.argmax(val_acc)
        ax.scatter(epochs[best_idx], val_acc[best_idx], c="red", s=60, zorder=5,
                   label=f"Best val: {val_acc[best_idx]:.2%}")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"{dataset} - Phase 2 Supervised Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f"  Saved {filename}")

def fig_threshold_scan():
    results = [
        ("1.0", 0.1135), ("0.5", 0.1135), ("0.2", 0.8873),
        ("0.1", 0.9116), ("0.05", 0.8895),
    ]
    labels, vals = zip(*results)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=COLORS["STDP"], edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("LIF Threshold", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("MNIST Pure STDP: Threshold Scan (20 STDP epochs)", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig3_mnist_threshold_scan.png")
    plt.close(fig)
    print("  Saved fig3_mnist_threshold_scan.png")

if __name__ == "__main__":
    print("Generating CIFAR-10 figures...")
    ann_c, snn_c, hnn_c = load_baseline_cifar10()
    pure_c, hybrid_c, pure_h_c, hybrid_h_c = collect_stdp_cifar10()
    fig_overview("CIFAR-10", ann_c, snn_c, hnn_c, pure_c, hybrid_c, "fig1_stdp_overview.png", 0.72)
    fig_curves("CIFAR-10", pure_h_c, hybrid_h_c, "fig2_stdp_curves.png")

    print("Generating MNIST figures...")
    ann_m, snn_m, hnn_m = load_baseline_mnist()
    pure_m, hybrid_m, pure_h_m, hybrid_h_m = collect_stdp_mnist()
    fig_overview("MNIST", ann_m, snn_m, hnn_m, pure_m, hybrid_m, "fig4_mnist_overview.png", 1.05)
    fig_curves("MNIST", pure_h_m, hybrid_h_m, "fig5_mnist_curves.png")
    fig_threshold_scan()

    print("Done!")
