import torch
from pathlib import Path

from evaluation.metrics import calculate_metrics


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        checkpoint_dir="results/checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0.0

        for x, y in loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x)

            loss = self.criterion(pred, y)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):

        self.model.eval()

        predictions = []
        targets = []

        for x, y in loader:

            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)

            predictions.append(pred.cpu())
            targets.append(y.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        return calculate_metrics(predictions, targets)

    def fit(self, train_loader, val_loader, epochs):

        best_val = float("inf")

        for epoch in range(1, epochs + 1):

            train_loss = self.train_epoch(train_loader)

            val_metrics = self.evaluate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MSE: {val_metrics['MSE']:.6f} | "
                f"Val MAE: {val_metrics['MAE']:.6f}"
            )

            if val_metrics["MSE"] < best_val:

                best_val = val_metrics["MSE"]

                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / "best.pt",
                )

        return best_val