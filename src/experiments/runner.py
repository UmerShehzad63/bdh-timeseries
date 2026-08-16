from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TSLIB_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Time-Series-Library"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIG
# ============================================================

DATASETS = {
    "ETTh1": {
        "data": "ETTh1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "variables": 7,
    },
    "ETTh2": {
        "data": "ETTh2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "variables": 7,
    },
    "ETTm1": {
        "data": "ETTm1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "variables": 7,
    },
    "ETTm2": {
        "data": "ETTm2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "variables": 7,
    },
    "Exchange": {
        "data": "custom",
        "root_path": "./dataset/exchange_rate/",
        "data_path": "exchange_rate.csv",
        "variables": 8,
    },
    "Weather": {
        "data": "custom",
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "variables": 21,
    },
    "Electricity": {
        "data": "custom",
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "variables": 321,
    },
    "Traffic": {
        "data": "custom",
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "variables": 862,
    },
}


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "DLinear",
    "PatchTST",
}


# ============================================================
# RUNNER
# ============================================================

def run_tslib(
    model: str,
    dataset: str,
    pred_len: int,
    seq_len: int = 96,
    batch_size: int = 32,
    train_epochs: int = 1,
    learning_rate: float | None = None,
):
    """
    Run one baseline experiment using Time-Series-Library.

    IMPORTANT:
    The measured `wall_time_sec` is the complete process runtime.

    It is NOT labelled as pure CUDA computation time.

    True CUDA-event training time must be measured inside the model's
    training process and will be added to the collection pipeline
    separately.
    """

    if model not in MODELS:
        raise ValueError(
            f"Unsupported model: {model}. "
            f"Supported models: {sorted(MODELS)}"
        )

    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {sorted(DATASETS)}"
        )

    if not TSLIB_ROOT.exists():
        raise FileNotFoundError(
            f"Time-Series-Library not found:\n{TSLIB_ROOT}"
        )

    cfg = DATASETS[dataset]
    variables = cfg["variables"]

    # --------------------------------------------------------
    # Base command
    # --------------------------------------------------------

    command = [
        sys.executable,
        "-u",
        "run.py",

        "--task_name",
        "long_term_forecast",

        "--is_training",
        "1",

        "--root_path",
        cfg["root_path"],

        "--data_path",
        cfg["data_path"],

        "--model_id",
        f"{dataset}_{seq_len}_{pred_len}",

        "--model",
        model,

        "--data",
        cfg["data"],

        "--features",
        "M",

        "--seq_len",
        str(seq_len),

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
        str(train_epochs),

        "--batch_size",
        str(batch_size),
    ]

    if learning_rate is not None:
        command.extend(
            [
                "--learning_rate",
                str(learning_rate),
            ]
        )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = os.environ.copy()

    # Ensure UTF-8 output on Windows.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # Make CUDA explicit when available.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    run_name = f"{model}_{dataset}_{seq_len}_{pred_len}"

    log_path = LOGS_DIR / f"{run_name}.log"
    result_path = RESULTS_DIR / f"{run_name}.json"

    print("=" * 70)
    print("TSLIB EXPERIMENT")
    print("=" * 70)
    print(f"Model:          {model}")
    print(f"Dataset:        {dataset}")
    print(f"Sequence:       {seq_len}")
    print(f"Horizon:        {pred_len}")
    print(f"Variables:      {variables}")
    print(f"Batch size:     {batch_size}")
    print(f"Epochs:         {train_epochs}")
    print(f"TSLib root:     {TSLIB_ROOT}")
    print()
    print("Command:")
    print(" ".join(command))
    print("=" * 70)

    # --------------------------------------------------------
    # ACTUAL PROCESS RUNTIME
    # --------------------------------------------------------
    #
    # This is intentionally kept separate from inference time.
    # It measures the complete training process from launch to
    # process completion.
    #

    start = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=TSLIB_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )

    end = time.perf_counter()

    wall_time_sec = end - start

    output = completed.stdout

    # --------------------------------------------------------
    # Save raw log
    # --------------------------------------------------------

    log_path.write_text(
        output,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Extract final MSE / MAE from TSLib output
    # --------------------------------------------------------

    mse = None
    mae = None

    for line in reversed(output.splitlines()):

        line = line.strip()

        if "mse:" in line.lower() and "mae:" in line.lower():

            try:
                parts = line.split(",")

                for part in parts:

                    key, value = part.split(":")

                    key = key.strip().lower()
                    value = value.strip()

                    if key == "mse":
                        mse = float(value)

                    elif key == "mae":
                        mae = float(value)

                break

            except Exception:
                pass

    # --------------------------------------------------------
    # Result record
    # --------------------------------------------------------

    result = {
        "model": model,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "variables": variables,
        "batch_size": batch_size,
        "train_epochs": train_epochs,
        "learning_rate": learning_rate,

        # Evaluation
        "test_mse": mse,
        "test_mae": mae,

        # Runtime
        "wall_time_sec": wall_time_sec,

        # This will be populated once CUDA-event timing is
        # instrumented inside the training process.
        "gpu_compute_time_sec": None,

        # Deliberately separate.
        "inference_time_sec": None,

        "return_code": completed.returncode,
        "log_file": str(log_path),
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)

    if completed.returncode == 0:
        print("EXPERIMENT COMPLETE")
    else:
        print("EXPERIMENT FAILED")

    print("=" * 70)

    print(f"Test MSE:           {mse}")
    print(f"Test MAE:           {mae}")
    print(f"Wall time:          {wall_time_sec:.3f} sec")
    print(f"GPU compute time:   {None}")
    print(f"Result:             {result_path}")
    print(f"Log:                {log_path}")

    print("=" * 70)

    if completed.returncode != 0:
        print()
        print("----- TSLib OUTPUT -----")
        print(output)
        print("------------------------")

        raise RuntimeError(
            f"{model} failed on {dataset}, horizon={pred_len}"
        )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # First experiment only.
    #
    # Once this succeeds, we expand systematically.

    run_tslib(
        model="DLinear",
        dataset="ETTh1",
        pred_len=96,
        seq_len=96,
        batch_size=32,
        train_epochs=1,
    )