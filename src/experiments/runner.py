from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[2]
TSLIB = ROOT / "data" / "raw" / "Time-Series-Library"

RESULTS_DIR = ROOT / "results" / "metrics"


def dataset_args(dataset: str):
    mapping = {
        "ETTh1": ("ETTh1.csv", "ETTh1", 7, "ETT-small"),
        "ETTh2": ("ETTh2.csv", "ETTh2", 7, "ETT-small"),
        "ETTm1": ("ETTm1.csv", "ETTm1", 7, "ETT-small"),
        "ETTm2": ("ETTm2.csv", "ETTm2", 7, "ETT-small"),
        "Electricity": ("electricity.csv", "custom", 321, "electricity"),
        "Exchange": ("exchange_rate.csv", "custom", 8, "exchange_rate"),
        "Traffic": ("traffic.csv", "custom", 862, "traffic"),
        "Weather": ("weather.csv", "custom", 21, "weather"),
    }

    if dataset not in mapping:
        raise ValueError(f"Unknown dataset: {dataset}")

    return mapping[dataset]


def build_command(
    model: str,
    dataset: str,
    pred_len: int,
    epochs: int = 10,
    batch_size: int = 32,
):
    filename, data_name, variables, folder = dataset_args(dataset)

    root_path = f"./dataset/{folder}/"

    cmd = [
    sys.executable,
    "-u",
    "run.py",
    "--task_name",
    "long_term_forecast",
    "--is_training",
    "1",
    "--root_path",
    root_path,
    "--data_path",
    filename,
    "--model_id",
    f"{dataset}_96_{pred_len}",
    "--model",
    model,
    "--data",
    data_name,
    "--features",
    "M",
    "--seq_len",
    "96",
    "--label_len",
    "48",
    "--pred_len",
    str(pred_len),
    "--enc_in",
    str(variables),
    "--dec_in",
    str(variables),
    "--c_out",
    str(variables),
    "--des",
    "Research",
    "--itr",
    "1",
    "--train_epochs",
    str(epochs),
    "--batch_size",
    str(batch_size),
]

    return cmd


def run_tslib(
    model: str,
    dataset: str,
    pred_len: int,
    epochs: int = 10,
    batch_size: int = 32,
):
    cmd = build_command(
        model=model,
        dataset=dataset,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("=" * 70)
    print("TSLIB EXPERIMENT")
    print("=" * 70)
    print("Model:", model)
    print("Dataset:", dataset)
    print("Horizon:", pred_len)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 70)

    completed = subprocess.run(
        cmd,
        cwd=TSLIB,
        text=True,
        capture_output=True,
    )

    print(completed.stdout)

    if completed.returncode != 0:
        print(completed.stderr)
        raise RuntimeError(
            f"{model} failed on {dataset}, horizon={pred_len}"
        )

    return completed.stdout


def parse_metrics(output: str):
    """
    Extract the final:
        mse:...
        mae:...
    reported by TSLib.
    """

    matches = re.findall(
        r"mse:([0-9eE.+-]+),\s*mae:([0-9eE.+-]+)",
        output,
    )

    if not matches:
        raise ValueError("Could not find mse/mae in TSLib output.")

    mse, mae = matches[-1]

    mse = float(mse)
    mae = float(mae)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": mse ** 0.5,
    }


def save_result(
    model: str,
    dataset: str,
    pred_len: int,
    metrics: dict,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{model.lower().replace('-', '_')}_"
        f"{dataset}_{pred_len}_seed42.json"
    )

    path = RESULTS_DIR / filename

    result = {
        "model": model,
        "dataset": dataset,
        "seq_len": 96,
        "pred_len": pred_len,
        "seed": 42,
        "metrics": metrics,
    }

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Saved:", path)

    return path


if __name__ == "__main__":
    # Local smoke test.
    output = run_tslib(
        model="DLinear",
        dataset="ETTh1",
        pred_len=96,
        epochs=1,
        batch_size=32,
    )

    metrics = parse_metrics(output)

    print("Parsed metrics:")
    print(metrics)

    save_result(
        model="DLinear",
        dataset="ETTh1",
        pred_len=96,
        metrics=metrics,
    )