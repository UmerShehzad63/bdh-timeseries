import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, pred_len=96, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.projection = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        output, (h, c) = self.lstm(x)

        last = output[:, -1:, :]
        last = last.expand(-1, self.pred_len, -1)

        return self.projection(last)


LSTMForecaster = LSTMModel