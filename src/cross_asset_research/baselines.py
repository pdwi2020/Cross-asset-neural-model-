"""Baseline cross-asset forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = x.T @ x
    ridge = 1e-6 * np.eye(xtx.shape[0])
    return np.linalg.pinv(xtx + ridge) @ x.T @ y


def _safe_std(x: np.ndarray, floor: float = 1e-4) -> float:
    if len(x) <= 2:
        return max(floor, 0.05)
    s = float(np.std(x, ddof=1))
    if not np.isfinite(s):
        return max(floor, 0.05)
    return max(s, floor)


def _har_cols(asset: str, *, include_jump: bool) -> List[str]:
    cols = [
        f"{asset}_log_rv_lag1",
        f"{asset}_log_rv_wavg",
        f"{asset}_log_rv_mavg",
    ]
    if include_jump:
        cols.append(f"{asset}_jump_lag1")
    return cols


@dataclass
class ModelForecast:
    """Forecast payload for one model on one split."""

    mu: pd.DataFrame
    sigma: pd.DataFrame
    distribution: str = "gaussian"
    dof: pd.DataFrame | None = None


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
            self.resid_std_[asset] = _safe_std(resid)

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            mu[asset] = test_df[f"{asset}_log_rv_lag1"].values
            sigma[asset] = self.resid_std_.get(asset, 0.05)
        return ModelForecast(mu=mu, sigma=sigma)


class HARModel:
    """Univariate HAR-RV regression for each asset."""

    name = "har_rv"
    include_jump = False

    def fit(self, train_df: pd.DataFrame, assets: Iterable[str]) -> None:
        self.assets = list(assets)
        self.coef_: Dict[str, np.ndarray] = {}
        self.resid_std_: Dict[str, float] = {}

        for asset in self.assets:
            cols = _har_cols(asset, include_jump=self.include_jump)
            x = train_df[cols].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            y = train_df[f"log_{asset}_rv"].to_numpy(dtype=float)
            beta = _ols_fit(x, y)
            pred = x @ beta
            resid = y - pred
            self.coef_[asset] = beta
            self.resid_std_[asset] = _safe_std(resid)

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            cols = _har_cols(asset, include_jump=self.include_jump)
            x = test_df[cols].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            mu[asset] = x @ self.coef_[asset]
            sigma[asset] = self.resid_std_.get(asset, 0.05)
        return ModelForecast(mu=mu, sigma=sigma)


class HARJModel(HARModel):
    """HAR-J model with jump regressor."""

    name = "har_j_rv"
    include_jump = True


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
        self.resid_std_ = {a: _safe_std(resid[:, i]) for i, a in enumerate(self.assets)}

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        x_cols = [f"{a}_log_rv_lag1" for a in self.assets]
        x = test_df[x_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        mu_vals = x @ self.beta_

        mu = pd.DataFrame(mu_vals, index=test_df.index, columns=self.assets)
        sigma = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            sigma[asset] = self.resid_std_.get(asset, 0.05)
        return ModelForecast(mu=mu, sigma=sigma)


class GARCH11Model:
    """Gaussian QML GARCH(1,1) baseline mapped into log-RV space."""

    name = "garch11_qml"

    @staticmethod
    def _nll(params: np.ndarray, r: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12

        n = len(r)
        var0 = max(float(np.var(r)), 1e-8)
        sigma2 = np.empty(n, dtype=float)
        sigma2[0] = var0
        for t in range(1, n):
            sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        sigma2 = np.clip(sigma2, 1e-12, None)

        ll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (r**2) / sigma2)
        if not np.isfinite(ll):
            return 1e12
        return float(ll)

    def fit(self, train_df: pd.DataFrame, assets: Iterable[str]) -> None:
        self.assets = list(assets)
        self.params_: Dict[str, np.ndarray] = {}
        self.last_state_: Dict[str, tuple[float, float]] = {}
        self.bias_: Dict[str, float] = {}
        self.resid_std_: Dict[str, float] = {}

        for asset in self.assets:
            r_col = f"{asset}_ret"
            if r_col in train_df.columns:
                r = train_df[r_col].to_numpy(dtype=float)
            else:
                # Fallback for datasets without returns columns.
                y = train_df[f"log_{asset}_rv"].to_numpy(dtype=float)
                r = np.diff(np.concatenate([[y[0]], y]))

            r = r - float(np.mean(r))
            var_r = max(float(np.var(r)), 1e-8)

            init = np.array([0.05 * var_r, 0.08, 0.88], dtype=float)
            bounds = [(1e-12, 1.0), (1e-8, 0.999), (1e-8, 0.999)]
            cons = ({"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]})

            res = minimize(
                fun=self._nll,
                x0=init,
                args=(r,),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 300, "ftol": 1e-8, "disp": False},
            )
            params = res.x if res.success else init
            omega, alpha, beta = [float(v) for v in params]

            sigma2 = np.empty(len(r), dtype=float)
            sigma2[0] = var_r
            for t in range(1, len(r)):
                sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
            sigma2 = np.clip(sigma2, 1e-12, None)

            mu_train = np.log1p(sigma2)
            y_true = train_df[f"log_{asset}_rv"].to_numpy(dtype=float)
            bias = float(np.mean(y_true - mu_train))
            resid = y_true - (mu_train + bias)

            self.params_[asset] = np.array([omega, alpha, beta], dtype=float)
            self.last_state_[asset] = (float(r[-1] ** 2), float(sigma2[-1]))
            self.bias_[asset] = bias
            self.resid_std_[asset] = _safe_std(resid)

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        mu = pd.DataFrame(index=test_df.index)
        sigma = pd.DataFrame(index=test_df.index)

        for asset in self.assets:
            omega, alpha, beta = [float(v) for v in self.params_[asset]]
            prev_r2, prev_sigma2 = self.last_state_[asset]

            r_series = (
                test_df[f"{asset}_ret"].to_numpy(dtype=float)
                if f"{asset}_ret" in test_df.columns
                else np.zeros(len(test_df), dtype=float)
            )

            mu_asset = np.empty(len(test_df), dtype=float)
            for i in range(len(test_df)):
                sigma2_next = omega + alpha * prev_r2 + beta * prev_sigma2
                sigma2_next = max(float(sigma2_next), 1e-12)
                mu_asset[i] = np.log1p(sigma2_next) + self.bias_[asset]

                prev_r2 = float(r_series[i] ** 2)
                prev_sigma2 = sigma2_next

            mu[asset] = mu_asset
            sigma[asset] = self.resid_std_.get(asset, 0.05)

        return ModelForecast(mu=mu, sigma=sigma)


class ProbHARModel(HARJModel):
    """HAR-J variant explicitly treated as probabilistic Gaussian baseline."""

    name = "prob_har_gaussian"


class StudentTHARModel(HARJModel):
    """HAR-J baseline with Student-t predictive distribution."""

    name = "prob_har_student_t"

    def fit(self, train_df: pd.DataFrame, assets: Iterable[str]) -> None:
        super().fit(train_df, assets)
        self.dof_: Dict[str, float] = {}
        for asset in self.assets:
            cols = _har_cols(asset, include_jump=True)
            x = train_df[cols].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            y = train_df[f"log_{asset}_rv"].to_numpy(dtype=float)
            pred = x @ self.coef_[asset]
            resid = y - pred

            std = _safe_std(resid)
            z = resid / std
            m2 = float(np.mean(z**2))
            m4 = float(np.mean(z**4))
            if m2 <= 1e-10:
                dof = 30.0
            else:
                excess_kurt = max(m4 / (m2**2) - 3.0, 0.0)
                dof = 200.0 if excess_kurt <= 1e-6 else 6.0 / excess_kurt + 4.0
            self.dof_[asset] = float(np.clip(dof, 4.2, 200.0))

    def predict(self, test_df: pd.DataFrame) -> ModelForecast:
        base = super().predict(test_df)
        dof_df = pd.DataFrame(index=test_df.index)
        for asset in self.assets:
            dof_df[asset] = self.dof_.get(asset, 30.0)
        return ModelForecast(mu=base.mu, sigma=base.sigma, distribution="student_t", dof=dof_df)


def default_models(
    *,
    include_student_t: bool = True,
    include_garch: bool = True,
    include_har_j: bool = True,
) -> List[object]:
    """Default baseline model suite."""

    models: List[object] = [NaiveLastModel(), HARModel()]
    if include_har_j:
        models.append(HARJModel())
    models.append(VAR1Model())
    if include_garch:
        models.append(GARCH11Model())
    models.append(ProbHARModel())
    if include_student_t:
        models.append(StudentTHARModel())
    return models


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
