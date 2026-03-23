"""LSTM model for RUL prediction with Monte Carlo Dropout."""
import numpy as np
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """Two-layer LSTM with MC Dropout for uncertainty estimation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch,) RUL predictions
        """
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # last timestep
        out = self.relu(self.fc1(self.dropout(out)))
        return self.fc2(out).squeeze(-1)

    def predict_with_ci(
        self, x: torch.Tensor, n_passes: int = 50
    ) -> tuple:
        """Run MC Dropout inference.

        Returns:
            (mean_pred, lower_95, upper_95) as floats
        """
        self.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x).item())
        mean = float(np.mean(preds))
        std = float(np.std(preds))
        return mean, mean - 1.96 * std, mean + 1.96 * std


def save_model(model: LSTMPredictor, path: str) -> None:
    """Save model weights and config together."""
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config},
        path
    )


def load_model(path: str) -> LSTMPredictor:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model = LSTMPredictor(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
