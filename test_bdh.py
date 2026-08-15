import sys
sys.path.insert(0, "src")

import torch

from models.bdh.ts_bdh import TSBDH


model = TSBDH(
    input_dim=7,
    output_dim=7,
    seq_len=96,
    pred_len=96,
)

x = torch.randn(4, 96, 7)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)