"""Statistical comparison tests for forecast models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class DMResult:
    """Diebold-Mariano pairwise test result."""

    model_a: str
    model_b: str
    n_obs: int
    dm_stat: float
    p_value: float
    mean_loss_diff: float
    better_model: str


def _newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance estimate for sample mean scaling."""

    n = len(x)
    if n <= 1:
        return np.nan

    x = x - np.mean(x)
    gamma0 = float(np.dot(x, x) / n)
    if lag <= 0:
        return gamma0

    var = gamma0
    for l in range(1, min(lag + 1, n - 1)):
        weight = 1.0 - l / (lag + 1.0)
        cov = float(np.dot(x[l:], x[:-l]) / n)
        var += 2.0 * weight * cov
    return max(var, 1e-12)


def diebold_mariano_test(
    loss_a: Iterable[float],
    loss_b: Iterable[float],
    *,
    horizon: int = 1,
) -> DMResult:
    """Two-sided Diebold-Mariano test using normal approximation."""

    la = np.asarray(list(loss_a), dtype=float)
    lb = np.asarray(list(loss_b), dtype=float)
    mask = np.isfinite(la) & np.isfinite(lb)
    la = la[mask]
    lb = lb[mask]

    if len(la) != len(lb):
        raise ValueError("loss vectors must have equal length after filtering")
    if len(la) < 8:
        raise ValueError("at least 8 paired observations are required")

    d = la - lb
    n = len(d)
    lag = max(int(horizon) - 1, 0)
    var_d = _newey_west_variance(d, lag)
    mean_d = float(np.mean(d))
    if not np.isfinite(var_d) or var_d <= 1e-12 or not np.isfinite(mean_d):
        dm_stat = 0.0
        p_value = 1.0
    else:
        dm_stat = float(mean_d / np.sqrt(var_d / n))
        p_value = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))

    if mean_d < 0:
        better = "model_a"
    elif mean_d > 0:
        better = "model_b"
    else:
        better = "tie"

    return DMResult(
        model_a="model_a",
        model_b="model_b",
        n_obs=n,
        dm_stat=dm_stat,
        p_value=p_value,
        mean_loss_diff=mean_d,
        better_model=better,
    )


def holm_adjust(p_values: Dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down multiple-comparison correction."""

    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)

    rows: List[dict] = []
    accepted = True
    for i, (name, p) in enumerate(items, start=1):
        threshold = alpha / (m - i + 1)
        reject = accepted and (p <= threshold)
        if not reject:
            accepted = False
        rows.append(
            {
                "hypothesis": name,
                "raw_p": float(p),
                "holm_threshold": float(threshold),
                "reject_h0": bool(reject),
            }
        )

    return pd.DataFrame(rows)


def benjamini_hochberg_adjust(p_values: Dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR control."""

    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)

    max_rank = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            max_rank = i

    rows: List[dict] = []
    for i, (name, p) in enumerate(items, start=1):
        rows.append(
            {
                "hypothesis": name,
                "raw_p": float(p),
                "bh_threshold": float((i / m) * alpha),
                "reject_h0": bool(i <= max_rank),
            }
        )
    return pd.DataFrame(rows)


def pairwise_dm_vs_baseline(
    losses_by_model: Dict[str, pd.Series],
    *,
    baseline: str,
    horizon: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run DM tests comparing each model to a baseline model."""

    if baseline not in losses_by_model:
        raise ValueError(f"baseline model {baseline!r} not found")

    baseline_loss = losses_by_model[baseline].sort_index()
    rows: List[dict] = []
    p_map: Dict[str, float] = {}

    for model, series in losses_by_model.items():
        if model == baseline:
            continue

        aligned = pd.concat([baseline_loss, series.sort_index()], axis=1, join="inner").dropna()
        if len(aligned) < 8:
            continue

        dm = diebold_mariano_test(aligned.iloc[:, 1].to_numpy(), aligned.iloc[:, 0].to_numpy(), horizon=horizon)
        p_map[f"{model}_vs_{baseline}"] = dm.p_value
        rows.append(
            {
                "model": model,
                "baseline": baseline,
                "n_obs": dm.n_obs,
                "dm_stat": dm.dm_stat,
                "p_value": dm.p_value,
                "mean_loss_diff_model_minus_baseline": dm.mean_loss_diff,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    holm = holm_adjust(p_map, alpha=alpha).rename(
        columns={
            "hypothesis": "comparison",
            "raw_p": "holm_raw_p",
            "reject_h0": "holm_reject_h0",
        }
    )
    bh = benjamini_hochberg_adjust(p_map, alpha=alpha).rename(
        columns={
            "hypothesis": "comparison",
            "raw_p": "bh_raw_p",
            "reject_h0": "bh_reject_h0",
        }
    )

    out["comparison"] = out["model"] + "_vs_" + out["baseline"]
    out = out.merge(
        holm[["comparison", "holm_threshold", "holm_reject_h0"]],
        on="comparison",
        how="left",
    )
    out = out.merge(
        bh[["comparison", "bh_threshold", "bh_reject_h0"]],
        on="comparison",
        how="left",
    )

    return out.sort_values("p_value").reset_index(drop=True)
