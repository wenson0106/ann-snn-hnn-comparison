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

def load_baseline():
    import sys
    sys.path.insert(0, str(Path(".").resolve()))
    ROOT = Path("runs")
    def load_run(path):
        if not path.exists():
            return None
        test = json.load(open(path / "test.json"))
        args = json.load(open(path / "args.json"))
        try:
            history = json.load(open(path / "history.json"))
            best = max(history, key=lambda h: h["val"]["acc"])
        except:
            best = {"epoch": 0}
        return test, best, args

    snn = load_run(ROOT / "snn_cifar10" / "20260526-145103")
    ann = load_run(ROOT / "ann_cifar10" / "20260526-144318")
    hnn = load_run(ROOT / "hnn_cifar10" / "20260526-151814")
    return ann, snn, hnn

def collect_stdp():
    pure_dir = sorted(Path("runs_stdp/stdp_pure").iterdir())[-1]
    hybrid_dir = sorted(Path("runs_stdp/stdp_hybrid").iterdir())[-1]
    pure_test = json.load(open(pure_dir / "test.json"))
    hybrid_test = json.load(open(hybrid_dir / "test.json"))
    pure_history = json.load(open(pure_dir / "history.json"))
    hybrid_history = json.load(open(hybrid_dir / "history.json"))
    return pure_test, hybrid_test, pure_history, hybrid_history

def fig_stdp_overview():
    ann, snn, hnn = load_baseline()
    pure_test, hybrid_test, pure_hist, hybrid_hist = collect_stdp()

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
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("CIFAR-10: STDP Experiments vs BP Baselines", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig1_stdp_overview.png")
    plt.close(fig)
    print("  Saved fig1_stdp_overview.png")

def fig_stdp_curves():
    pure_dir = sorted(Path("runs_stdp/stdp_pure").iterdir())[-1]
    hybrid_dir = sorted(Path("runs_stdp/stdp_hybrid").iterdir())[-1]
    pure_hist = json.load(open(pure_dir / "history.json"))
    hybrid_hist = json.load(open(hybrid_dir / "history.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    datasets = [
        (ax1, pure_hist, "Pure STDP (readout BP)", COLORS["STDP"],
         [h["train_acc"] for h in pure_hist], [h["val_acc"] for h in pure_hist]),
        (ax2, hybrid_hist, "Hybrid STDP+BP", COLORS["Hybrid"],
         [h["train"]["acc"] for h in hybrid_hist], [h["val"]["acc"] for h in hybrid_hist]),
    ]

    for ax, hist, title, color, train_acc, val_acc in datasets:
        epochs = [h["epoch"] for h in hist]
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

    fig.suptitle("Phase 2 Supervised Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig2_stdp_curves.png")
    plt.close(fig)
    print("  Saved fig2_stdp_curves.png")

if __name__ == "__main__":
    print("Generating STDP figures...")
    fig_stdp_overview()
    fig_stdp_curves()
    print("Done!")
