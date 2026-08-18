from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import random
import numpy as np

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TSLIB_ROOT = PROJECT_ROOT / "data" / "raw" / "Time-Series-Library"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIG
# ============================================================

DATASETS = {
    "ETTh1": {"data": "ETTh1", "root_path": "./dataset/ETT-small/", "data_path": "ETTh1.csv", "variables": 7},
    "ETTh2": {"data": "ETTh2", "root_path": "./dataset/ETT-small/", "data_path": "ETTh2.csv", "variables": 7},
    "ETTm1": {"data": "ETTm1", "root_path": "./dataset/ETT-small/", "data_path": "ETTm1.csv", "variables": 7},
    "ETTm2": {"data": "ETTm2", "root_path": "./dataset/ETT-small/", "data_path": "ETTm2.csv", "variables": 7},
    "Exchange": {"data": "custom", "root_path": "./dataset/exchange_rate/", "data_path": "exchange_rate.csv", "variables": 8},
    "Weather": {"data": "custom", "root_path": "./dataset/weather/", "data_path": "weather.csv", "variables": 21},
    "Electricity": {"data": "custom", "root_path": "./dataset/electricity/", "data_path": "electricity.csv", "variables": 321},
    "Traffic": {"data": "custom", "root_path": "./dataset/traffic/", "data_path": "traffic.csv", "variables": 862},
}

MODELS = {"DLinear", "PatchTST", "LSTM"}


# ============================================================
# RESULT COLLECTION
# ============================================================

def update_summary(result: dict) -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    row = {
        "model": result["model"],
        "dataset": result["dataset"],
        "seq_len": result["seq_len"],
        "pred_len": result["pred_len"],
        "variables": result["variables"],
        "batch_size": result["batch_size"],
        "train_epochs": result["train_epochs"],
        "learning_rate": result.get("learning_rate"),
        "test_mse": result.get("test_mse"),
        "test_mae": result.get("test_mae"),
        "test_rmse": result.get("test_rmse"),
        "wall_time_sec": result.get("wall_time_sec"),
        "gpu_compute_time_sec": result.get("gpu_compute_time_sec"),
        "inference_time_sec": result.get("inference_time_sec"),
        "trainable_parameters": result.get("trainable_parameters"),
        "return_code": result.get("return_code", 0),
    }
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        key = (df["model"] == row["model"]) & (df["dataset"] == row["dataset"]) & (df["seq_len"] == row["seq_len"]) & (df["pred_len"] == row["pred_len"])
        df = df.loc[~key]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(summary_path, index=False)


# ============================================================
# TSLIB BASELINES
# ============================================================

def run_tslib(model: str, dataset: str, pred_len: int, seq_len: int = 96, batch_size: int = 32, train_epochs: int = 1, learning_rate: float | None = None):
    if model not in {"DLinear", "PatchTST"}:
        raise ValueError(f"TSLib model must be DLinear or PatchTST, got {model}")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not TSLIB_ROOT.exists():
        raise FileNotFoundError(f"Time-Series-Library not found: {TSLIB_ROOT}")

    cfg = DATASETS[dataset]
    variables = cfg["variables"]
    command = [sys.executable, "-u", "run.py", "--task_name", "long_term_forecast", "--is_training", "1", "--root_path", cfg["root_path"], "--data_path", cfg["data_path"], "--model_id", f"{dataset}_{seq_len}_{pred_len}", "--model", model, "--data", cfg["data"], "--features", "M", "--seq_len", str(seq_len), "--label_len", "48", "--pred_len", str(pred_len), "--enc_in", str(variables), "--dec_in", str(variables), "--c_out", str(variables), "--des", "Research", "--itr", "1", "--train_epochs", str(train_epochs), "--batch_size", str(batch_size)]
    if learning_rate is not None:
        command += ["--learning_rate", str(learning_rate)]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

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

    start = time.perf_counter()
    completed = subprocess.run(command, cwd=TSLIB_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    wall_time_sec = time.perf_counter() - start
    output = completed.stdout
    log_path.write_text(output, encoding="utf-8")

    mse = mae = None
    for line in reversed(output.splitlines()):
        if "mse:" in line.lower() and "mae:" in line.lower():
            try:
                values = {part.split(":", 1)[0].strip().lower(): float(part.split(":", 1)[1].strip()) for part in line.split(",")}
                mse, mae = values.get("mse"), values.get("mae")
                break
            except Exception:
                pass

    result = {"model": model, "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len, "variables": variables, "batch_size": batch_size, "train_epochs": train_epochs, "learning_rate": learning_rate, "test_mse": mse, "test_mae": mae, "test_rmse": float(np.sqrt(mse)) if mse is not None else None, "wall_time_sec": wall_time_sec, "gpu_compute_time_sec": None, "inference_time_sec": None, "trainable_parameters": None, "return_code": completed.returncode, "log_file": str(log_path)}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    update_summary(result)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE" if completed.returncode == 0 else "EXPERIMENT FAILED")
    print("=" * 70)
    print(f"Test MSE:           {mse}")
    print(f"Test MAE:           {mae}")
    print(f"Wall time:          {wall_time_sec:.3f} sec")
    print(f"GPU compute time:   None (not instrumented inside TSLib)")
    print(f"Result:             {result_path}")
    print(f"Summary CSV:        {RESULTS_DIR / 'summary.csv'}")
    print("=" * 70)
    if completed.returncode != 0:
        print(output)
        raise RuntimeError(f"{model} failed on {dataset}, horizon={pred_len}")
    return result


# ============================================================
# LSTM BASELINE
# ============================================================

class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.values[idx:idx + self.seq_len]
        y = self.values[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, pred_len: int = 96):
        super().__init__()
        self.pred_len = pred_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, pred_len * input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.head(out[:, -1])
        return out.view(x.size(0), self.pred_len, self.input_size)


def run_lstm(dataset: str, pred_len: int, seq_len: int = 96, batch_size: int = 32, train_epochs: int = 1, learning_rate: float = 1e-3, hidden_size: int = 128, num_layers: int = 2):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg = DATASETS[dataset]
    data_path = PROJECT_ROOT / "data" / "raw" / ("ETT-small" if dataset.startswith("ETT") else "TSLib-datasets")
    if dataset.startswith("ETT"):
        data_path = data_path / cfg["data_path"]
    else:
        folder = {"Exchange": "exchange_rate", "Weather": "weather", "Electricity": "electricity", "Traffic": "traffic"}[dataset]
        data_path = data_path / folder / cfg["data_path"]

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variables = cfg["variables"]
    df = pd.read_csv(data_path)
    values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if values.shape[1] != variables:
        values = values[:, -variables:]

    n = len(values)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train_raw = values[:train_end]
    val_raw = values[train_end:val_end]
    test_raw = values[val_end:]

    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = (values - mean) / std

    # Use the same 60/20/20 target split while allowing each validation/test
    # window to use the historical context immediately preceding its target.
    train = normalized[:train_end]
    val_context = normalized[train_end - seq_len:val_end]
    test_context = normalized[val_end - seq_len:]

    train_ds = WindowDataset(train, seq_len, pred_len)
    val_ds = WindowDataset(val_context, seq_len, pred_len)
    test_ds = WindowDataset(test_context, seq_len, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = LSTMForecaster(variables, hidden_size, num_layers, 0.1, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 70)
    print("LSTM TIME-SERIES EXPERIMENT")
    print("=" * 70)
    print(f"Device:              {device}")
    if device.type == "cuda":
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")
    print(f"Dataset:             {dataset}")
    print(f"Sequence length:     {seq_len}")
    print(f"Prediction length:   {pred_len}")
    print(f"Batch size:          {batch_size}")
    print(f"Epochs:              {train_epochs}")
    print(f"Learning rate:       {learning_rate}")
    print(f"Variables:           {variables}")
    print(f"Train samples:       {len(train_ds)}")
    print(f"Val samples:         {len(val_ds)}")
    print(f"Test samples:        {len(test_ds)}")
    print(f"Trainable parameters: {params}")

    start_wall = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    gpu_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if gpu_start is not None:
        gpu_start.record()

    best_val = float("inf")
    for epoch in range(1, train_epochs + 1):
        model.train()
        train_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * x.size(0)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                val_sum += criterion(model(x), y).item() * x.size(0)
        train_loss = train_sum / len(train_ds)
        val_loss = val_sum / len(val_ds)
        print(f"Epoch {epoch:02d}/{train_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), RESULTS_DIR / f"LSTM_{dataset}_{seq_len}_{pred_len}_best.pt")

    if gpu_end is not None:
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_compute_time_sec = gpu_start.elapsed_time(gpu_end) / 1000.0
    else:
        gpu_compute_time_sec = None
    wall_time_sec = time.perf_counter() - start_wall

    model.load_state_dict(torch.load(RESULTS_DIR / f"LSTM_{dataset}_{seq_len}_{pred_len}_best.pt", map_location=device, weights_only=True))
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu().numpy()
            predictions.append(pred)
            targets.append(y.numpy())
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(mse))

    result = {"model": "LSTM", "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len, "variables": variables, "batch_size": batch_size, "train_epochs": train_epochs, "learning_rate": learning_rate, "hidden_size": hidden_size, "num_layers": num_layers, "trainable_parameters": params, "best_val_mse": best_val, "test_mse": mse, "test_mae": mae, "test_rmse": rmse, "wall_time_sec": wall_time_sec, "gpu_compute_time_sec": gpu_compute_time_sec, "inference_time_sec": None, "return_code": 0}
    result_path = RESULTS_DIR / f"LSTM_{dataset}_{seq_len}_{pred_len}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    update_summary(result)

    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Best validation MSE: {best_val}")
    print(f"Test MSE:            {mse}")
    print(f"Test MAE:            {mae}")
    print(f"Test RMSE:           {rmse}")
    print(f"Wall time:           {wall_time_sec:.3f} sec")
    print(f"GPU compute time:    {gpu_compute_time_sec:.3f} sec" if gpu_compute_time_sec is not None else "GPU compute time:    None")
    print(f"Result:              {result_path}")
    print(f"Summary CSV:         {RESULTS_DIR / 'summary.csv'}")
    print("=" * 70)
    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ETTh1")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    if args.model == "LSTM":
        run_lstm(args.dataset, args.pred_len, args.seq_len, args.batch_size, args.train_epochs, args.learning_rate or 1e-3)
    else:
        run_tslib(args.model, args.dataset, args.pred_len, args.seq_len, args.batch_size, args.train_epochs, args.learning_rate)