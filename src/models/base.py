import torch
import torch.nn as nn


class ForecastModel(nn.Module):
    """
    Common interface for all forecasting models.

    Input:
        x: [batch, context_length, variables]

    Output:
        y_hat: [batch, prediction_length, variables]
    """

    def __init__(self):
        super().__init__()

    def forecast(self, x):
        raise NotImplementedError