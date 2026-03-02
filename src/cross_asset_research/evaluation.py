"""Evaluation metrics for point and probabilistic forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import norm, t


@dataclass
class AssetMetric:
    """Per-asset summary metrics."""

    asset: str
    rmse: float
    mae: float
    nll: float
    coverage_95: float
    avg_interval_width_95: float


@dataclass
class ModelEvaluation:
    """Model-wide evaluation payload."""

    model: str
    by_asset: List[AssetMetric]
    aggregate_rmse: float
    aggregate_mae: float
    aggregate_nll: float
    aggregate_coverage_95: float
    losses: Dict[str, List[float]]


def evaluate_model(
    *,
    model_name: str,
    y_true: pd.DataFrame,
    mu_pred: pd.DataFrame,
    sigma_pred: pd.DataFrame,
    distribution: str = "gaussian",
    dof_pred: pd.DataFrame | None = None,
    assets: Iterable[str],
) -> ModelEvaluation:
    """Evaluate point and probabilistic quality for one model."""

    assets = list(assets)
    out: List[AssetMetric] = []
    losses: Dict[str, List[float]] = {}

    for asset in assets:
        y = y_true[asset].to_numpy(dtype=float)
        m = mu_pred[asset].to_numpy(dtype=float)
        s = np.maximum(sigma_pred[asset].to_numpy(dtype=float), 1e-6)

        err = y - m
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))

        if distribution == "student_t" and dof_pred is not None and asset in dof_pred.columns:
            nu = np.maximum(dof_pred[asset].to_numpy(dtype=float), 2.1)
            z = t.ppf(0.975, df=nu)
            z = np.asarray(z, dtype=float)
            standardized = err / s
            nll_series = -(t.logpdf(standardized, df=nu) - np.log(s))
            lower = m - z * s
            upper = m + z * s
        else:
            # Gaussian NLL.
            nll_series = 0.5 * np.log(2.0 * np.pi * (s**2)) + 0.5 * ((err / s) ** 2)
            z = norm.ppf(0.975)
            lower = m - z * s
            upper = m + z * s
        nll = float(np.mean(nll_series))
        cover = float(np.mean((y >= lower) & (y <= upper)))
        width = float(np.mean(upper - lower))

        out.append(
            AssetMetric(
                asset=asset,
                rmse=rmse,
                mae=mae,
                nll=nll,
                coverage_95=cover,
                avg_interval_width_95=width,
            )
        )
        losses[asset] = [float(x) for x in (err**2).tolist()]

    agg_rmse = float(np.mean([m.rmse for m in out]))
    agg_mae = float(np.mean([m.mae for m in out]))
    agg_nll = float(np.mean([m.nll for m in out]))
    agg_cov = float(np.mean([m.coverage_95 for m in out]))

    return ModelEvaluation(
        model=model_name,
        by_asset=out,
        aggregate_rmse=agg_rmse,
        aggregate_mae=agg_mae,
        aggregate_nll=agg_nll,
        aggregate_coverage_95=agg_cov,
        losses=losses,
    )


def evaluations_to_frame(evals: Iterable[ModelEvaluation]) -> pd.DataFrame:
    """Tabular leaderboard from model evaluation objects."""

    rows: List[dict] = []
    for ev in evals:
        rows.append(
            {
                "model": ev.model,
                "aggregate_rmse": ev.aggregate_rmse,
                "aggregate_mae": ev.aggregate_mae,
                "aggregate_nll": ev.aggregate_nll,
                "aggregate_coverage_95": ev.aggregate_coverage_95,
            }
        )
    return pd.DataFrame(rows).sort_values("aggregate_rmse", ascending=True).reset_index(drop=True)
