"""
Neural Network Residual Correction Layer

The physics model (RC network) captures the dominant thermal dynamics
but misses real-world effects: occupancy patterns, solar gain variability,
HVAC transients, infiltration changes.

This neural network learns the RESIDUAL between physics predictions
and actual measurements. The hybrid model is:

    T_predicted = T_physics + NN_correction(features)

This is the same hybrid pattern used in the CSH2 digital twin
(physics engine + PINN residual model).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional


class ResidualCorrectionNet(nn.Module):
    """MLP that predicts the residual between RC model and reality.

    Input features:
    - Hour of day, day of week (cyclical encoded)
    - Outside temperature, solar radiation
    - RC model predicted temperature
    - Lagged residuals (autoregressive component)

    Output: scalar correction to add to RC model prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def encode_cyclical(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode a cyclical feature (hour, day-of-week) as sin/cos pair.

    This prevents the model from seeing hour=23 and hour=0 as far apart.
    """
    sin_vals = np.sin(2 * np.pi * values / period)
    cos_vals = np.cos(2 * np.pi * values / period)
    return sin_vals, cos_vals


def build_correction_features(
    df: pd.DataFrame,
    T_physics: np.ndarray,
    T_measured: np.ndarray,
    n_lags: int = 3,
) -> pd.DataFrame:
    """Build the feature matrix for the correction network.

    Args:
        df: Raw data with weather and time columns
        T_physics: RC model predicted indoor temperature
        T_measured: Actual measured indoor temperature
        n_lags: Number of lagged residual features

    Returns:
        Feature DataFrame ready for training
    """
    features = {}

    # Time features (cyclical)
    if isinstance(df.index, pd.DatetimeIndex):
        hour_sin, hour_cos = encode_cyclical(df.index.hour.values, 24.0)
        dow_sin, dow_cos = encode_cyclical(df.index.dayofweek.values, 7.0)
        features["hour_sin"] = hour_sin
        features["hour_cos"] = hour_cos
        features["dow_sin"] = dow_sin
        features["dow_cos"] = dow_cos

    # Physics model prediction as a feature
    features["T_physics"] = T_physics

    # Weather features (pass through)
    weather_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["temp", "solar", "wind", "humid", "cloud"]
    )]
    for col in weather_cols:
        features[col] = df[col].values

    # Lagged residuals (autoregressive)
    residual = T_measured - T_physics
    for lag in range(1, n_lags + 1):
        features[f"residual_lag_{lag}"] = np.roll(residual, lag)
        features[f"residual_lag_{lag}"][:lag] = 0.0

    return pd.DataFrame(features, index=df.index)


class CorrectionTrainer:
    """Training loop for the residual correction network."""

    def __init__(
        self,
        model: ResidualCorrectionNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 256,
    ) -> float:
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            self.optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_val)
            loss = self.criterion(pred, y_val).item()
        self.val_losses.append(loss)
        return loss
