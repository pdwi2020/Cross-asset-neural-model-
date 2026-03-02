"""Risk backtesting utilities for probabilistic volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


@dataclass
class AssetRiskBacktest:
    """Per-asset VaR/ES backtest summary."""

    model: str
    asset: str
    alpha: float
    n_obs: int
    expected_exceed_rate: float
    observed_exceed_rate: float
    exceedances: int
    kupiec_lr_uc: float
    kupiec_p_value: float
    christoffersen_lr_ind: float
    christoffersen_p_value: float
    avg_var_level: float
    avg_es_level: float
    avg_excess_given_exceedance: float


def gaussian_var_es(mu: np.ndarray, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper-tail Gaussian VaR and ES for volatility/risk exceedance events."""

    z = norm.ppf(alpha)
    pdf_z = norm.pdf(z)
    tail_prob = max(1.0 - alpha, 1e-6)
    var = mu + z * sigma
    es = mu + (pdf_z / tail_prob) * sigma
    return var, es


def kupiec_unconditional_coverage_test(exceed: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec LR-UC test for exceedance frequency calibration."""

    n = int(len(exceed))
    if n == 0:
        return float("nan"), float("nan")

    x = int(np.sum(exceed))
    p0 = max(1.0 - alpha, 1e-9)
    phat = min(max(x / n, 1e-9), 1.0 - 1e-9)

    ll_null = (n - x) * np.log(1.0 - p0) + x * np.log(p0)
    ll_alt = (n - x) * np.log(1.0 - phat) + x * np.log(phat)
    lr = float(-2.0 * (ll_null - ll_alt))
    p = float(1.0 - chi2.cdf(lr, df=1))
    return lr, p


def christoffersen_independence_test(exceed: np.ndarray) -> tuple[float, float]:
    """Christoffersen LR-independence test for exceedance clustering."""

    if len(exceed) < 3:
        return float("nan"), float("nan")

    x = exceed.astype(int)
    x_prev = x[:-1]
    x_curr = x[1:]

    n00 = int(np.sum((x_prev == 0) & (x_curr == 0)))
    n01 = int(np.sum((x_prev == 0) & (x_curr == 1)))
    n10 = int(np.sum((x_prev == 1) & (x_curr == 0)))
    n11 = int(np.sum((x_prev == 1) & (x_curr == 1)))

    denom0 = max(n00 + n01, 1)
    denom1 = max(n10 + n11, 1)
    pi0 = min(max(n01 / denom0, 1e-9), 1 - 1e-9)
    pi1 = min(max(n11 / denom1, 1e-9), 1 - 1e-9)
    pi = min(max((n01 + n11) / max(n00 + n01 + n10 + n11, 1), 1e-9), 1 - 1e-9)

    ll_ind = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_dep = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)

    lr = float(-2.0 * (ll_ind - ll_dep))
    p = float(1.0 - chi2.cdf(lr, df=1))
    return lr, p


def run_var_es_backtest(
    *,
    model_name: str,
    y_true: pd.DataFrame,
    mu_pred: pd.DataFrame,
    sigma_pred: pd.DataFrame,
    assets: Iterable[str],
    alpha: float = 0.95,
) -> pd.DataFrame:
    """Run VaR/ES calibration diagnostics for all assets."""

    rows: List[dict] = []
    assets = list(assets)

    for asset in assets:
        y = y_true[asset].to_numpy(dtype=float)
        mu = mu_pred[asset].to_numpy(dtype=float)
        sigma = np.maximum(sigma_pred[asset].to_numpy(dtype=float), 1e-6)

        var, es = gaussian_var_es(mu, sigma, alpha)
        exceed = y > var

        lr_uc, p_uc = kupiec_unconditional_coverage_test(exceed, alpha)
        lr_ind, p_ind = christoffersen_independence_test(exceed)

        excess = np.maximum(y - var, 0.0)
        avg_excess = float(excess[exceed].mean()) if np.any(exceed) else 0.0

        rows.append(
            AssetRiskBacktest(
                model=model_name,
                asset=asset,
                alpha=float(alpha),
                n_obs=int(len(y)),
                expected_exceed_rate=float(1.0 - alpha),
                observed_exceed_rate=float(np.mean(exceed)),
                exceedances=int(np.sum(exceed)),
                kupiec_lr_uc=lr_uc,
                kupiec_p_value=p_uc,
                christoffersen_lr_ind=lr_ind,
                christoffersen_p_value=p_ind,
                avg_var_level=float(np.mean(var)),
                avg_es_level=float(np.mean(es)),
                avg_excess_given_exceedance=avg_excess,
            ).__dict__
        )

    return pd.DataFrame(rows)
