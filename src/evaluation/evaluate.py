import json
from pathlib import Path

from evaluation.metrics import calculate_metrics


def evaluate_model(model, loader, device, output_path=None):

    model.eval()

    predictions = []
    targets = []

    import torch

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            predictions.append(pred.cpu())
            targets.append(y.cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    metrics = calculate_metrics(predictions, targets)

    if output_path is not None:

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics