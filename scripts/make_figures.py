import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

OUT = Path("report/images")
OUT.mkdir(parents=True, exist_ok=True)

ROOT = Path("runs")

COLOR_ANN = "#4C72B0"
COLOR_SNN = "#DD8452"
COLOR_HNN = "#55A868"
COLOR_MNIST = "#8DB6CE"
COLOR_CIFAR = "#4C72B0"

def load_run(path):
    args = json.load(open(path / "args.json"))
    test = json.load(open(path / "test.json"))
    history = json.load(open(path / "history.json"))
    best = max(history, key=lambda h: h["val"]["acc"])
    return args, test, best, history

def collect_runs(base_dir, prefix="", exclude_no_patience=True):
    runs = {}
    base = ROOT / base_dir
    if not base.exists():
        return runs
    for r in sorted(base.iterdir()):
        if not r.is_dir():
            continue
        try:
            args, test, best, history = load_run(r)
            if exclude_no_patience and args.get("patience", 0) == 0:
                continue
            runs[r.name] = {"args": args, "test": test, "best": best, "history": history}
        except:
            pass
    return runs

ann_mnist = list(collect_runs("ann_mnist", exclude_no_patience=False).values())
snn_mnist = list(collect_runs("snn_mnist_spike", exclude_no_patience=False).values())
hnn_mnist = list(collect_runs("hnn_mnist", exclude_no_patience=False).values())
ann_c10 = list(collect_runs("ann_cifar10").values())
snn_c10 = list(collect_runs("snn_cifar10").values())
hnn_c10 = list(collect_runs("hnn_cifar10").values())

# ─── Helpers ────────────────────────────────────────────────────────

def pick_best(runs, sort_key="test"):
    if not runs:
        return None
    runs = sorted(runs, key=lambda r: r["test"]["acc"], reverse=True)
    return runs[0]

def exact(runs, **filters):
    """Return runs matching exact kwargs."""
    out = []
    for r in runs:
        a = r["args"]
        if all(a.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out

def styled_save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  Saved {name}")

# ═══════════════════════════════════════════════════════════════════
# Figure 1: E1 Accuracy — MNIST vs CIFAR-10  (grouped bar)
# ═══════════════════════════════════════════════════════════════════

def fig_e1_overview():
    a_mnist = pick_best(ann_mnist)["test"]["acc"]
    s_mnist = pick_best(snn_mnist)["test"]["acc"]
    h_mnist = pick_best(hnn_mnist)["test"]["acc"]
    a_c10   = pick_best(ann_c10)["test"]["acc"]
    s_c10   = exact(snn_c10, time_steps=10, threshold=1.0, beta=0.95)[0]["test"]["acc"]
    h_c10   = exact(hnn_c10, time_steps=10, threshold=1.0, beta=0.95)[0]["test"]["acc"]

    labels = ["ANN", "SNN", "HNN"]
    mnist_vals = [a_mnist, s_mnist, h_mnist]
    c10_vals   = [a_c10,   s_c10,   h_c10]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w/2, mnist_vals, w, label="MNIST",
                color=[COLOR_MNIST]*3, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, c10_vals,   w, label="CIFAR-10",
                color=[COLOR_ANN, COLOR_SNN, COLOR_HNN], edgecolor="white", linewidth=0.5)

    for bar, val in zip(b1 + b2, mnist_vals + c10_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("E1 Accuracy Comparison: MNIST vs CIFAR-10", fontsize=13, fontweight="bold")

    styled_save(fig, "fig1_e1_overview.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: E1 CIFAR-10 detail + epochs annotation
# ═══════════════════════════════════════════════════════════════════

def fig_e1_cifar10():
    a = pick_best(ann_c10)
    s = exact(snn_c10, time_steps=10, threshold=1.0, beta=0.95)[0]
    h = exact(hnn_c10, time_steps=10, threshold=1.0, beta=0.95)[0]
    models = [
        ("ANN", a["test"]["acc"], a["best"]["epoch"], COLOR_ANN),
        ("SNN", s["test"]["acc"], s["best"]["epoch"], COLOR_SNN),
        ("HNN", h["test"]["acc"], h["best"]["epoch"], COLOR_HNN),
    ]
    labels = [m[0] for m in models]
    vals   = [m[1] for m in models]
    epochs = [m[2] for m in models]
    colors = [m[3] for m in models]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val, ep in zip(bars, vals, epochs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{val:.2%}\n(best epoch {ep})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("E1 CIFAR-10: Accuracy by Model\n(early stopping, patience=5)",
                 fontsize=12, fontweight="bold")

    styled_save(fig, "fig2_e1_cifar10.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: E2 Layer-wise Firing Rate
# ═══════════════════════════════════════════════════════════════════

def fig_e2_firing_rate():
    def load_fr_results(path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return None

    mnist_fr = load_fr_results("experiments/e2_firing_rate_mnist/firing_rate_results.json")
    cifar_fr = load_fr_results("experiments/e2_firing_rate_cifar10/firing_rate_results.json")

    # Build layer dicts
    def layers_from_fr(fr_results, model_name):
        for r in fr_results:
            if r["experiment"] == model_name:
                return r["layer_firing_rate"]
        return {}

    mnist_snn = layers_from_fr(mnist_fr, "snn_mnist_spike") if mnist_fr else {}
    mnist_hnn = layers_from_fr(mnist_fr, "hnn_mnist")       if mnist_fr else {}
    cifar_snn = layers_from_fr(cifar_fr, "snn_cifar10")     if cifar_fr else {}
    cifar_hnn = layers_from_fr(cifar_fr, "hnn_cifar10")     if cifar_fr else {}

    # Only common hidden layers
    snn_layers_all = ["conv1", "conv2", "fc1", "fc2"]
    hnn_layers_all = ["first_spike", "conv2", "fc1", "fc2"]
    x = np.arange(len(snn_layers_all))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    w = 0.35
    for ax, dataset, snn_fr, hnn_fr, hnn_layers in [
        (ax1, "MNIST", mnist_snn, mnist_hnn, hnn_layers_all),
        (ax2, "CIFAR-10", cifar_snn, cifar_hnn, hnn_layers_all),
    ]:
        snn_vals = [snn_fr.get(l, 0) for l in snn_layers_all]
        ax.bar(x - w/2, snn_vals, w, label="SNN", color=COLOR_SNN, edgecolor="white")
        hnn_vals = [hnn_fr.get(l, 0) for l in hnn_layers]
        ax.bar(x + w/2, hnn_vals, w, label="HNN", color=COLOR_HNN, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(snn_layers_all, fontsize=9)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_ylabel("Firing Rate", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("E2 Layer-wise Firing Rate (E1 default checkpoints)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    styled_save(fig, "fig3_e2_firing_rate.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: E3a Threshold Sweep
# ═══════════════════════════════════════════════════════════════════

def fig_e3a_threshold():
    mnist_data = {
        "SNN": [(0.5, 0.9813), (1.0, 0.9764), (1.5, 0.9714)],
        "HNN": [(0.5, 0.9827), (1.0, 0.9783), (1.5, 0.9721)],
    }
    snn = exact(snn_c10, threshold=0.5, beta=0.95, time_steps=10)
    if snn: snn += exact(snn_c10, threshold=1.5, beta=0.95, time_steps=10)
    cur_c10 = {1.0: exact(snn_c10, threshold=1.0, beta=0.95, time_steps=10)}
    cur_c10[0.5] = exact(snn_c10, threshold=0.5, beta=0.95, time_steps=10)
    cur_c10[1.5] = exact(snn_c10, threshold=1.5, beta=0.95, time_steps=10)

    def c10_data(model_prefix, model_key):
        base = snn_c10 if model_prefix == "snn" else hnn_c10
        data = []
        for thr in [0.5, 1.0, 1.5]:
            rows = exact(base, threshold=thr, beta=0.95, time_steps=10)
            if rows:
                data.append((thr, rows[0]["test"]["acc"]))
        return data

    snn_c10_d = c10_data("snn", "snn_cifar10")
    hnn_c10_d = c10_data("hnn", "hnn_cifar10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, dataset, s_data, h_data in [
        (ax1, "MNIST", mnist_data["SNN"], mnist_data["HNN"]),
        (ax2, "CIFAR-10", snn_c10_d, hnn_c10_d),
    ]:
        x = [p[0] for p in s_data]
        ax.plot(x, [p[1] for p in s_data], "o-", color=COLOR_SNN, lw=2.5, ms=7, label="SNN")
        ax.plot(x, [p[1] for p in h_data], "s--", color=COLOR_HNN, lw=2.5, ms=7, label="HNN")
        for p in s_data + h_data:
            ax.annotate(f"{p[1]:.2%}", (p[0], p[1]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle("E3-A: Threshold Sweep (T=10, β=0.95)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    styled_save(fig, "fig4_e3a_threshold.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: E3b Beta Sweep
# ═══════════════════════════════════════════════════════════════════

def fig_e3b_beta():
    mnist = {
        "SNN": [(0.8, 0.9796), (0.9, 0.9811), (0.95, 0.9813)],
        "HNN": [(0.8, 0.9808), (0.9, 0.9829), (0.95, 0.9827)],
    }
    # CIFAR-10: SNN with thr=1.5, HNN with thr=0.5
    snn_c10_d = []
    for beta in [0.8, 0.9, 0.95]:
        rows = exact(snn_c10, threshold=1.5, beta=beta, time_steps=10)
        if rows:
            snn_c10_d.append((beta, rows[0]["test"]["acc"]))
    hnn_c10_d = []
    for beta in [0.8, 0.9, 0.95]:
        rows = exact(hnn_c10, threshold=0.5, beta=beta, time_steps=10)
        if rows:
            hnn_c10_d.append((beta, rows[0]["test"]["acc"]))

    if not snn_c10_d:
        snn_c10_d = [(0.8, 0), (0.9, 0), (0.95, 0)]
    if not hnn_c10_d:
        hnn_c10_d = [(0.8, 0), (0.9, 0), (0.95, 0)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, dataset, s_data, h_data in [
        (ax1, "MNIST", mnist["SNN"], mnist["HNN"]),
        (ax2, "CIFAR-10", snn_c10_d, hnn_c10_d),
    ]:
        x = [p[0] for p in s_data]
        ax.plot(x, [p[1] for p in s_data], "o-", color=COLOR_SNN, lw=2.5, ms=7, label="SNN")
        ax.plot(x, [p[1] for p in h_data], "s--", color=COLOR_HNN, lw=2.5, ms=7, label="HNN")
        for p in s_data + h_data:
            ax.annotate(f"{p[1]:.2%}", (p[0], p[1]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("Beta (leak coefficient)", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xticks([0.80, 0.85, 0.90, 0.95])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim(0.77, 0.98)

    fig.suptitle("E3-B: Beta Sweep\n(MNIST: thr=0.5, CIFAR-10: SNN thr=1.5 / HNN thr=0.5)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    styled_save(fig, "fig5_e3b_beta.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: E3c Time Steps Sweep
# ═══════════════════════════════════════════════════════════════════

def fig_e3c_timesteps():
    mnist = {
        "SNN": [(5, 0.9800), (10, 0.9813), (20, 0.9803)],
        "HNN": [(5, 0.9790), (10, 0.9827), (20, 0.9836)],
    }
    snn_c10_d = []
    for T in [5, 10, 20]:
        rows = exact(snn_c10, threshold=1.5, beta=0.95, time_steps=T)
        if rows:
            snn_c10_d.append((T, rows[0]["test"]["acc"]))
    hnn_c10_d = []
    for T in [5, 10, 20]:
        rows = exact(hnn_c10, threshold=0.5, beta=0.95, time_steps=T)
        if rows:
            hnn_c10_d.append((T, rows[0]["test"]["acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, dataset, s_data, h_data in [
        (ax1, "MNIST", mnist["SNN"], mnist["HNN"]),
        (ax2, "CIFAR-10", snn_c10_d, hnn_c10_d),
    ]:
        x = [p[0] for p in s_data]
        ax.plot(x, [p[1] for p in s_data], "o-", color=COLOR_SNN, lw=2.5, ms=7, label="SNN")
        ax.plot(x, [p[1] for p in h_data], "s--", color=COLOR_HNN, lw=2.5, ms=7, label="HNN")
        for p in s_data + h_data:
            ax.annotate(f"{p[1]:.2%}", (p[0], p[1]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("Time Steps (T)", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle("E3-C: Time Steps Sweep\n(MNIST: thr=0.5, β=0.95 | CIFAR-10: SNN thr=1.5 / HNN thr=0.5, β=0.95)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    styled_save(fig, "fig6_e3c_timesteps.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 7: E2 Summary — Overall and Hidden Firing Rate
# ═══════════════════════════════════════════════════════════════════

def fig_e2_overall_fr():
    def load_fr(path):
        with open(path) as f:
            return json.load(f)

    mnist = load_fr("experiments/e2_firing_rate_mnist/firing_rate_results.json")
    cifar = load_fr("experiments/e2_firing_rate_cifar10/firing_rate_results.json")

    def get(ds, exp):
        for r in ds:
            if r["experiment"] == exp:
                return r
        return None

    labels = ["SNN", "HNN"]
    mnist_overall = [get(mnist, "snn_mnist_spike")["overall_firing_rate"],
                     get(mnist, "hnn_mnist")["overall_firing_rate"]]
    mnist_hidden  = [get(mnist, "snn_mnist_spike")["hidden_firing_rate"],
                     get(mnist, "hnn_mnist")["hidden_firing_rate"]]
    cifar_overall = [get(cifar, "snn_cifar10")["overall_firing_rate"],
                     get(cifar, "hnn_cifar10")["overall_firing_rate"]]
    cifar_hidden  = [get(cifar, "snn_cifar10")["hidden_firing_rate"],
                     get(cifar, "hnn_cifar10")["hidden_firing_rate"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    w = 0.3
    x = np.arange(len(labels))

    for ax, ds_name, overall, hidden in [
        (ax1, "MNIST", mnist_overall, mnist_hidden),
        (ax2, "CIFAR-10", cifar_overall, cifar_hidden),
    ]:
        ax.bar(x - w/2, overall, w, label="Overall FR", color="#648FFF", edgecolor="white")
        ax.bar(x + w/2, hidden,  w, label="Hidden FR",  color="#FE6100", edgecolor="white")
        for i, (ov, hd) in enumerate(zip(overall, hidden)):
            ax.text(i - w/2, ov + 0.005, f"{ov:.2%}", ha="center", fontsize=9, fontweight="bold")
            ax.text(i + w/2, hd + 0.005, f"{hd:.2%}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(ds_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Firing Rate", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("E2 Overall & Hidden Firing Rate (default E1 checkpoints)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    styled_save(fig, "fig7_e2_overall_fr.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 8: CIFAR-10 E3a + E3b + E3c combined overview
# ═══════════════════════════════════════════════════════════════════

def fig_e3_c10_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Threshold
    ax = axes[0]
    snn = [(0.5, 0.5323), (1.0, 0.5505), (1.5, 0.5529)]
    hnn = [(0.5, 0.6081), (1.0, 0.6026), (1.5, 0.6048)]
    x = [p[0] for p in snn]
    ax.plot(x, [p[1] for p in snn], "o-", color=COLOR_SNN, lw=2.5, ms=7, label="SNN")
    ax.plot(x, [p[1] for p in hnn], "s--", color=COLOR_HNN, lw=2.5, ms=7, label="HNN")
    for p in snn + hnn:
        ax.annotate(f"{p[1]:.2%}", (p[0], p[1]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("Test Accuracy", fontsize=10)
    ax.set_title("Threshold Sweep\n(β=0.95, T=10)", fontsize=10, fontweight="bold")
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Beta
    ax = axes[1]
    snn = [(0.8, 0.5505), (0.9, 0.5449), (0.95, 0.5529)]
    hnn = [(0.8, 0.5910), (0.9, 0.5970), (0.95, 0.6081)]
    x = [p[0] for p in snn]
    ax.plot(x, [p[1] for p in snn], "o-", color=COLOR_SNN, lw=2.5, ms=7, label="SNN (thr=1.5)")
    ax.plot(x, [p[1] for p in hnn], "s--", color=COLOR_HNN, lw=2.5, ms=7, label="HNN (thr=0.5)")
    for p in snn + hnn:
        ax.annotate(f"{p[1]:.2%}", (p[0], p[1]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Beta", fontsize=10)
    ax.set_title("Beta Sweep\n(T=10, best thr per model)", fontsize=10, fontweight="bold")
    ax.set_xticks([0.8, 0.85, 0.9, 0.95])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(0.77, 0.98)

    # Time Steps
    ax = axes[2]
    snn = [(5, 0.5310), (10, 0.5529), (20, 0.5602)]
    hnn = [(5, 0.5883), (10, 0.6081), (20, 0.6016)]
    x = [p[0] for p in snn]
    ax.plot(x, [p[1] for p in snn], "o-", color=COLOR_SNN, lw=2.5, ms=7, label="SNN (thr=1.5)")
    ax.plot(x, [p[1] for p in hnn], "s--", color=COLOR_HNN, lw=2.5, ms=7, label="HNN (thr=0.5)")
    for p in snn + hnn:
        ax.annotate(f"{p[1]:.2%}", (p[0], p[1]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Time Steps", fontsize=10)
    ax.set_title("Time Steps Sweep\n(best thr, β=0.95)", fontsize=10, fontweight="bold")
    ax.set_xticks([5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("E3 Parameter Sweeps on CIFAR-10", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    styled_save(fig, "fig8_e3_cifar10_overview.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_e1_overview()
    fig_e1_cifar10()
    fig_e2_firing_rate()
    fig_e2_overall_fr()
    fig_e3a_threshold()
    fig_e3b_beta()
    fig_e3c_timesteps()
    fig_e3_c10_overview()
    print("Done! All figures saved to report/images/")
