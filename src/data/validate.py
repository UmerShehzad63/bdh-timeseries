import pandas as pd
from pathlib import Path

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

for name, path in DATASETS.items():
    df = pd.read_csv(Path(path))
    dates = pd.to_datetime(df["date"])

    print("=" * 60)
    print(name)
    print("Shape:", df.shape)
    print("Variables:", df.shape[1] - 1)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate timestamps:", int(dates.duplicated().sum()))
    print("Start:", dates.iloc[0])
    print("End:", dates.iloc[-1])

print("=" * 60)
print("VALIDATION COMPLETE")