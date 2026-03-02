"""Baseline cross-asset forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = x.T @ x
    ridge = 1e-6 * np.eye(xtx.shape[0])
    return np.linalg.pinv(xtx + ridge) @ x.T @ y


@dataclass
class ModelForecast:
    """Forecast payload for one model on one split."""

    mu: pd.DataFrame
    sigma: pd.DataFrame


class NaiveLastModel:
    """Predict next log-vol as previous log-vol."""

    name = "naive_last_surface"

    def fit(self, train_df: pd.DataFrame, assets: Iterable[str]) -> None:
        self.assets = list(assets)
        self.resid_std_: Dict[str, float] = {}
        for asset in self.assets:
            y = train_df[f"log_{asset}_rv"].to_numpy()
            pred = np.roll(y, 1)
            pred[0] = y[0]
            resid = y - pred
            self.resid_std_[asset] = float(np.std(resid, ddof=1) if len(resid) > 2 else 0.05)

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            mu[asset] = test_df[f"{asset}_log_rv_lag1"].values
            sigma[asset] = max(self.resid_std_.get(asset, 0.05), 1e-4)
        return ModelForecast(mu=mu, sigma=sigma)


class HARModel:
    """Univariate HAR-style regression for each asset."""

    name = "har_rv"

    def fit(self, train_df: pd.DataFrame, assets: Iterable[str]) -> None:
        self.assets = list(assets)
        self.coef_: Dict[str, np.ndarray] = {}
        self.resid_std_: Dict[str, float] = {}

        for asset in self.assets:
            cols = [
                f"{asset}_log_rv_lag1",
                f"{asset}_log_rv_wavg",
                f"{asset}_log_rv_mavg",
                f"{asset}_jump_lag1",
            ]
            x = train_df[cols].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            y = train_df[f"log_{asset}_rv"].to_numpy(dtype=float)
            beta = _ols_fit(x, y)
            pred = x @ beta
            resid = y - pred
            self.coef_[asset] = beta
            self.resid_std_[asset] = float(np.std(resid, ddof=1) if len(resid) > 2 else 0.05)

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            cols = [
                f"{asset}_log_rv_lag1",
                f"{asset}_log_rv_wavg",
                f"{asset}_log_rv_mavg",
                f"{asset}_jump_lag1",
            ]
            x = test_df[cols].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            mu[asset] = x @ self.coef_[asset]
            sigma[asset] = max(self.resid_std_.get(asset, 0.05), 1e-4)
        return ModelForecast(mu=mu, sigma=sigma)


class VAR1Model:
    """Simple VAR(1)-style cross-asset linear model on log RV states."""

    name = "var1_cross_asset"

    def fit(self, train_df: pd.DataFrame, assets: Iterable[str]) -> None:
        self.assets = list(assets)
        y_cols = [f"log_{a}_rv" for a in self.assets]
        x_cols = [f"{a}_log_rv_lag1" for a in self.assets]

        x = train_df[x_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        y = train_df[y_cols].to_numpy(dtype=float)
        beta = _ols_fit(x, y)
        pred = x @ beta
        resid = y - pred

        self.beta_ = beta
        self.resid_std_ = {
            a: float(np.std(resid[:, i], ddof=1) if len(resid) > 2 else 0.05) for i, a in enumerate(self.assets)
        }

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        x_cols = [f"{a}_log_rv_lag1" for a in self.assets]
        x = test_df[x_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        mu_vals = x @ self.beta_

        mu = pd.DataFrame(mu_vals, index=test_df.index, columns=self.assets)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            sigma[asset] = max(self.resid_std_.get(asset, 0.05), 1e-4)
        return ModelForecast(mu=mu, sigma=sigma)


class ProbHARModel(HARModel):
    """HAR variant explicitly treated as probabilistic baseline."""

    name = "prob_har_gaussian"


def default_models() -> List[object]:
    """Default model suite."""

    return [NaiveLastModel(), HARModel(), VAR1Model(), ProbHARModel()]


def fit_predict_models(
    models: Iterable[object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    assets: Iterable[str],
) -> Dict[str, ModelForecast]:
    """Fit and forecast for each model on one split."""

    assets = list(assets)
    out: Dict[str, ModelForecast] = {}
    for model in models:
        model.fit(train_df, assets)
        out[model.name] = model.predict(test_df)
    return out
