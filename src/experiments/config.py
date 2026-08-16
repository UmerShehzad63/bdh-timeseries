DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Exchange",
    "Traffic",
    "Weather",
]

HORIZONS = [96, 192, 336, 720]

SEQ_LEN = 96
LABEL_LEN = 48

MODELS = [
    "BDH",
    "S-Mamba",
    "DLinear",
    "PatchTST",
    "LSTM",
]

SEEDS = [42]

RESULTS_DIR = "results/metrics"


# Dataset metadata used by the experiment runner.
DATASET_INFO = {
    "ETTh1": {
        "data": "ETTh1",
        "path": "data/raw/ETT-small/ETTh1.csv",
        "variables": 7,
        "freq": "h",
    },
    "ETTh2": {
        "data": "ETTh2",
        "path": "data/raw/ETT-small/ETTh2.csv",
        "variables": 7,
        "freq": "h",
    },
    "ETTm1": {
        "data": "ETTm1",
        "path": "data/raw/ETT-small/ETTm1.csv",
        "variables": 7,
        "freq": "15min",
    },
    "ETTm2": {
        "data": "ETTm2",
        "path": "data/raw/ETT-small/ETTm2.csv",
        "variables": 7,
        "freq": "15min",
    },
    "Electricity": {
        "data": "custom",
        "path": "data/raw/TSLib-datasets/electricity/electricity.csv",
        "variables": 321,
        "freq": "h",
    },
    "Exchange": {
        "data": "custom",
        "path": "data/raw/TSLib-datasets/exchange_rate/exchange_rate.csv",
        "variables": 8,
        "freq": "d",
    },
    "Traffic": {
        "data": "custom",
        "path": "data/raw/TSLib-datasets/traffic/traffic.csv",
        "variables": 862,
        "freq": "h",
    },
    "Weather": {
        "data": "custom",
        "path": "data/raw/TSLib-datasets/weather/weather.csv",
        "variables": 21,
        "freq": "10min",
    },
}