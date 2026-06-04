#!/usr/bin/env python3
import subprocess
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_CMD = [sys.executable, str(ROOT / "scripts" / "train_stdp.py"),
            "--dataset", "mnist", "--experiment", "stdp_pure",
            "--disable-progress",
            "--stdp-epochs", "20", "--epochs", "100", "--patience", "10",
            "--output-root", "runs_stdp_mnist"]

THRESHOLDS = [1.0, 0.5, 0.2, 0.1, 0.05]

# Phase 2 HP candidates (used after best threshold is found)
STDP_LRS = [0.001, 0.01, 0.1]
A_RATIOS = [  # (A_plus, A_minus)
    (0.01, 0.01),
    (0.005, 0.01),
    (0.01, 0.005),
]


def run_and_get_acc(cmd, desc):
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")
    # Parse final test accuracy from output
    for line in result.stdout.splitlines():
        if "Final test accuracy:" in line:
            acc = float(line.split(":")[1].strip())
            print(f"  -> {desc}: test_acc={acc:.4f} ({elapsed:.0f}s)")
            return acc
    # Fallback: read test.json
    return None


def main():
    results = {"threshold_scan": [], "hp_scan": []}

    # Phase 1: Threshold scan
    print("\n========== PHASE 1: THRESHOLD SCAN ==========")
    threshold_results = {}
    for thr in THRESHOLDS:
        cmd = BASE_CMD + ["--threshold", str(thr)]
        acc = run_and_get_acc(cmd, f"Threshold={thr}")
        threshold_results[thr] = acc
        results["threshold_scan"].append({"threshold": thr, "test_acc": acc})

    best_thr = max(threshold_results, key=threshold_results.get)
    print(f"\nBest threshold: {best_thr} (acc={threshold_results[best_thr]:.4f})")
    results["best_threshold"] = best_thr

    # Phase 2: HP scan at best threshold
    print(f"\n========== PHASE 2: HP SCAN (threshold={best_thr}) ==========")
    hp_results = []
    for stdp_lr in STDP_LRS:
        for A_plus, A_minus in A_RATIOS:
            cmd = BASE_CMD + ["--threshold", str(best_thr),
                              "--stdp-lr", str(stdp_lr),
                              "--A-plus", str(A_plus),
                              "--A-minus", str(A_minus),
                              "--stdp-epochs", "50"]
            acc = run_and_get_acc(cmd, f"lr={stdp_lr} A+={A_plus} A-={A_minus}")
            hp_results.append({
                "stdp_lr": stdp_lr, "A_plus": A_plus, "A_minus": A_minus, "test_acc": acc
            })
            results["hp_scan"].append(hp_results[-1])

    # Save summary
    summary_path = ROOT / "runs_stdp_mnist" / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep summary saved to {summary_path}")

    # Print summary table
    print("\n\n========== SWEEP SUMMARY ==========")
    print("\nThreshold scan:")
    for r in results["threshold_scan"]:
        print(f"  thr={r['threshold']:.2f} -> {r['test_acc']:.4f}")
    print(f"\nBest threshold: {best_thr} (acc={threshold_results[best_thr]:.4f})")
    print("\nHP scan:")
    for r in results["hp_scan"]:
        print(f"  lr={r['stdp_lr']:.4f} A+={r['A_plus']:.4f} A-={r['A_minus']:.4f} -> {r['test_acc']:.4f}")


if __name__ == "__main__":
    main()
