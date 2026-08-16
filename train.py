import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/data")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import LongHorizonDataset
from models.bdh.ts_bdh import TSBDH
from utils.seed import set_seed


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("BDH TIME-SERIES EXPERIMENT")
    print("=" * 70)

    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Sequence length:", args.seq_len)
    print("Prediction length:", args.pred_len)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Seed:", args.seed)

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    train_dataset = LongHorizonDataset(
        args.dataset,
        "train",
        args.seq_len,
        args.pred_len,
    )

    val_dataset = LongHorizonDataset(
        args.dataset,
        "val",
        args.seq_len,
        args.pred_len,
    )

    test_dataset = LongHorizonDataset(
        args.dataset,
        "test",
        args.seq_len,
        args.pred_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    variables = train_dataset.data.shape[1]

    print("Variables:", variables)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = TSBDH(
        input_dim=variables,
        output_dim=variables,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        D=128,
        H=4,
        N=4096,
        L=2,
        dropout=0.1,
    ).to(device)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", trainable_params)

    # --------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"bdh_{args.dataset}_{args.pred_len}.pt"
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):

        model.train()

        train_loss = 0.0
        train_samples = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            batch_size = x.size(0)

            train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss /= train_samples

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        # --------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset": args.dataset,
                    "seq_len": args.seq_len,
                    "pred_len": args.pred_len,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

            print("  -> Best model saved.")

        else:
            patience_counter += 1

            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start_time

    # --------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            total_squared_error += (
                (pred - y) ** 2
            ).sum().item()

            total_absolute_error += (
                torch.abs(pred - y)
            ).sum().item()

            total_elements += y.numel()

    test_mse = (
        total_squared_error
        / total_elements
    )

    test_mae = (
        total_absolute_error
        / total_elements
    )

    test_rmse = test_mse ** 0.5

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    results = {
        "model": "BDH",
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "seed": args.seed,
        "parameters": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "training_time_seconds": training_time,
    }

    result_dir = Path("results/metrics")
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        result_dir
        / f"bdh_{args.dataset}_{args.pred_len}_seed{args.seed}.json"
    )

    with open(result_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print("Best epoch:", best_epoch)
    print("Best validation MSE:", best_val_loss)
    print("Test MSE:", test_mse)
    print("Test MAE:", test_mae)
    print("Test RMSE:", test_rmse)
    print("Training time:", training_time, "seconds")
    print("Results:", result_path)


if __name__ == "__main__":
    main()