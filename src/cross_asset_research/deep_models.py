"""Neural probabilistic models for cross-asset volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from .baselines import ModelForecast
from .config import ModelConfig

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ProbabilisticLSTM(nn.Module):
        """LSTM predicting mean and scale for Gaussian forecasts."""

        def __init__(self, input_dim: int, hidden: int, layers: int, output_dim: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, output_dim * 2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out, _ = self.lstm(x)
            h = self.dropout(out[:, -1, :])
            p = self.head(h)
            half = p.shape[1] // 2
            mu = p[:, :half]
            sigma = torch.nn.functional.softplus(p[:, half:]) + 1e-4
            return mu, sigma

else:

    class ProbabilisticLSTM:  # pragma: no cover
        """Placeholder when torch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is not available. Install extras: pip install .[deep]")


@dataclass
class LSTMTrainResult:
    """Training diagnostics for probabilistic LSTM."""

    epochs_run: int
    best_val_nll: float


def _make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= seq_len:
        return np.empty((0, seq_len, x.shape[1])), np.empty((0, y.shape[1]))

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(len(x) - seq_len):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(xs), np.asarray(ys)


def _nll(y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y_true).sum(dim=1).mean()


def probabilistic_lstm_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    assets: Sequence[str],
    cfg: ModelConfig,
    *,
    seed: int = 42,
) -> tuple[ModelForecast, LSTMTrainResult | None]:
    """Train a probabilistic LSTM on one split and forecast test horizon."""

    if not TORCH_AVAILABLE:
        # Fallback: deterministic mean from lag1 + broad sigma.
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in assets:
            mu[asset] = test_df[f"{asset}_log_rv_lag1"].values
            sigma[asset] = 0.2
        return ModelForecast(mu=mu, sigma=sigma), None

    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train = train_df[list(feature_cols)].to_numpy(dtype=np.float32)
    y_train = train_df[list(target_cols)].to_numpy(dtype=np.float32)
    x_test = test_df[list(feature_cols)].to_numpy(dtype=np.float32)

    # Standardize using train statistics.
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-6

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_test_n = (x_test - x_mean) / x_std

    x_seq, y_seq = _make_sequences(x_train_n, y_train_n, cfg.sequence_length)
    if len(x_seq) < max(16, cfg.batch_size):
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)
        for i, asset in enumerate(assets):
            mu[asset] = np.repeat(float(y_mean[0, i]), len(test_df))
            sigma[asset] = np.repeat(float(y_std[0, i]), len(test_df))
        return ModelForecast(mu=mu, sigma=sigma), None

    # Small in-sample split for early stopping.
    n_total = len(x_seq)
    n_val = max(int(0.15 * n_total), 8)
    train_x_t = torch.tensor(x_seq[:-n_val], dtype=torch.float32)
    train_y_t = torch.tensor(y_seq[:-n_val], dtype=torch.float32)
    val_x_t = torch.tensor(x_seq[-n_val:], dtype=torch.float32)
    val_y_t = torch.tensor(y_seq[-n_val:], dtype=torch.float32)

    model = ProbabilisticLSTM(
        input_dim=train_x_t.shape[2],
        hidden=cfg.lstm_hidden,
        layers=cfg.lstm_layers,
        output_dim=len(target_cols),
        dropout=cfg.lstm_dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loader = DataLoader(TensorDataset(train_x_t, train_y_t), batch_size=cfg.batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    patience = 3
    bad = 0

    for epoch in range(cfg.lstm_epochs):
        model.train()
        for xb, yb in loader:
            mu, sigma = model(xb)
            loss = _nll(yb, mu, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mu_v, sigma_v = model(val_x_t)
            val_loss = float(_nll(val_y_t, mu_v, sigma_v).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Rolling autoregressive inference on test window.
    model.eval()
    hist_x = np.concatenate([x_train_n[-cfg.sequence_length :], x_test_n], axis=0)
    mu_out: List[np.ndarray] = []
    sigma_out: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(test_df)):
            seq = hist_x[i : i + cfg.sequence_length]
            seq_t = torch.tensor(seq[None, :, :], dtype=torch.float32)
            mu_n, sigma_n = model(seq_t)
            mu_np = mu_n.numpy()[0]
            sigma_np = sigma_n.numpy()[0]
            mu_real = mu_np * y_std[0] + y_mean[0]
            sigma_real = np.maximum(sigma_np * y_std[0], 1e-4)
            mu_out.append(mu_real)
            sigma_out.append(sigma_real)

    mu_df = pd.DataFrame(np.asarray(mu_out), index=test_df.index, columns=list(assets))
    sigma_df = pd.DataFrame(np.asarray(sigma_out), index=test_df.index, columns=list(assets))
    return ModelForecast(mu=mu_df, sigma=sigma_df), LSTMTrainResult(epochs_run=cfg.lstm_epochs, best_val_nll=best_val)
