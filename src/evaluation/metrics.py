import torch


def mse(pred, target):
    return torch.mean((pred - target) ** 2).item()


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def calculate_metrics(pred, target):
    return {
        "MSE": mse(pred, target),
        "MAE": mae(pred, target),
        "RMSE": rmse(pred, target),
    }