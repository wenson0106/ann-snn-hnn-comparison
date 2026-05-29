import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> dict:
    args = load_json(run_dir / "args.json")
    history = load_json(run_dir / "history.json")
    test = load_json(run_dir / "test.json")
    best_epoch = max(history, key=lambda row: row["val"]["acc"])

    return {
        "experiment": args["experiment"],
        "run_dir": str(run_dir),
        "epochs": args["epochs"],
        "batch_size": args["batch_size"],
        "time_steps": args["time_steps"],
        "beta": args["beta"],
        "threshold": args["threshold"],
        "input_gain": args.get("input_gain", 1.0),
        "seed": args["seed"],
        "best_epoch": best_epoch["epoch"],
        "best_val_acc": best_epoch["val"]["acc"],
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "test_num_samples": test["num_samples"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    rows = [summarize_run(Path(run_dir)) for run_dir in args.runs]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    for row in rows:
        print(
            f"{row['experiment']}: "
            f"best_val_acc={row['best_val_acc']:.4f}, "
            f"test_acc={row['test_acc']:.4f}, "
            f"run={row['run_dir']}"
        )


if __name__ == "__main__":
    main()
