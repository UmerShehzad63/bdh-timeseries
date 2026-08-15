import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from normalization import StandardScaler


DATASETS = {
    "ETTh1": "data/raw/ETT-small/ETTh1.csv",
    "ETTh2": "data/raw/ETT-small/ETTh2.csv",
    "ETTm1": "data/raw/ETT-small/ETTm1.csv",
    "ETTm2": "data/raw/ETT-small/ETTm2.csv",
    "Electricity": "data/raw/TSLib-datasets/electricity/electricity.csv",
    "Exchange": "data/raw/TSLib-datasets/exchange_rate/exchange_rate.csv",
    "Traffic": "data/raw/TSLib-datasets/traffic/traffic.csv",
    "Weather": "data/raw/TSLib-datasets/weather/weather.csv",
}


class LongHorizonDataset(Dataset):

    def __init__(
        self,
        name,
        split="train",
        seq_len=96,
        pred_len=96,
    ):
        self.name = name
        self.split = split
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(DATASETS[name])

        # Remove exact duplicate timestamps.
        df = df.drop_duplicates(subset=["date"], keep="first")

        self.dates = pd.to_datetime(df["date"])
        data = df.iloc[:, 1:].values.astype(np.float32)

        n = len(data)

        # Standard benchmark splits
        if name.startswith("ETT"):
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)
        else:
            train_end = int(n * 0.7)
            val_end = int(n * 0.8)

        if split == "train":
            start = 0
            end = train_end
        elif split == "val":
            start = train_end
            end = val_end
        elif split == "test":
            start = val_end
            end = n
        else:
            raise ValueError(f"Unknown split: {split}")

        # Fit scaler ONLY on training data.
        scaler = StandardScaler()
        scaler.fit(data[:train_end])

        self.scaler = scaler
        self.data = scaler.transform(data)

        self.start = start
        self.end = end

        # Need context before validation/test boundary.
        if split != "train":
            self.start -= seq_len

        self.start = max(0, self.start)

    def __len__(self):
        return max(
            0,
            self.end - self.start - self.seq_len - self.pred_len + 1
        )

    def __getitem__(self, index):

        i = self.start + index

        x = self.data[
            i : i + self.seq_len
        ]

        y = self.data[
            i + self.seq_len :
            i + self.seq_len + self.pred_len
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )