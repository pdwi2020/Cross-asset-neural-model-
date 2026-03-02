"""Statistical comparison tests for forecast models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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


def _align_loss_frame(losses_by_model: Dict[str, pd.Series]) -> pd.DataFrame:
    frames = []
    for model, series in losses_by_model.items():
        frames.append(series.sort_index().rename(model))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, join="inner").dropna()
    return out


def _moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    block = max(int(block_size), 1)
    idx: List[int] = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        for k in range(block):
            idx.append((start + k) % n)
            if len(idx) >= n:
                break
    return np.asarray(idx[:n], dtype=int)


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


def spa_test(
    losses_by_model: Dict[str, pd.Series],
    *,
    benchmark: str,
    n_bootstrap: int = 300,
    block_size: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Simplified Hansen SPA test against a benchmark model."""

    aligned = _align_loss_frame(losses_by_model)
    if aligned.empty or benchmark not in aligned.columns:
        return pd.DataFrame()

    challengers = [c for c in aligned.columns if c != benchmark]
    if not challengers:
        return pd.DataFrame()

    bench = aligned[benchmark].to_numpy(dtype=float)
    challenger_losses = aligned[challengers].to_numpy(dtype=float)
    d = bench[:, None] - challenger_losses  # positive => challenger better

    n = d.shape[0]
    mean_d = d.mean(axis=0)
    sd_d = d.std(axis=0, ddof=1) + 1e-8
    t_stats = np.sqrt(n) * mean_d / sd_d
    t_obs = float(np.max(np.maximum(t_stats, 0.0)))

    # Recentering under null for SPA bootstrap.
    d_centered = d - np.maximum(mean_d, 0.0)[None, :]

    rng = np.random.default_rng(seed)
    t_boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = _moving_block_bootstrap_indices(n, block_size, rng)
        db = d_centered[idx]
        mu_b = db.mean(axis=0)
        sd_b = db.std(axis=0, ddof=1) + 1e-8
        tb = np.sqrt(n) * mu_b / sd_b
        t_boot[b] = float(np.max(tb))

    spa_p = float((1.0 + np.sum(t_boot >= t_obs)) / (n_bootstrap + 1.0))

    rows: List[dict] = []
    for j, model in enumerate(challengers):
        p_one_sided = float(1.0 - norm.cdf(t_stats[j]))
        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "n_obs": int(n),
                "mean_loss_improvement_vs_benchmark": float(mean_d[j]),
                "t_stat": float(t_stats[j]),
                "model_one_sided_p": p_one_sided,
                "spa_global_t": t_obs,
                "spa_global_p": spa_p,
            }
        )

    return pd.DataFrame(rows).sort_values("model_one_sided_p").reset_index(drop=True)


def model_confidence_set(
    losses_by_model: Dict[str, pd.Series],
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 300,
    block_size: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate Model Confidence Set by iterative elimination vs current best model."""

    aligned = _align_loss_frame(losses_by_model)
    if aligned.empty or aligned.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    active = list(aligned.columns)
    elimination_rows: List[dict] = []
    step = 1

    while len(active) > 1:
        loss = aligned[active]
        mean_loss = loss.mean(axis=0)
        best = str(mean_loss.idxmin())

        worst_model = None
        worst_p = 1.0
        worst_diff = 0.0

        for model in active:
            if model == best:
                continue

            diff = (loss[model] - loss[best]).to_numpy(dtype=float)
            obs = float(np.mean(diff))
            if obs <= 0.0:
                continue

            boot_means = np.empty(n_bootstrap, dtype=float)
            for b in range(n_bootstrap):
                idx = _moving_block_bootstrap_indices(len(diff), block_size, rng)
                boot_means[b] = float(np.mean(diff[idx]))

            p_val = float((1.0 + np.sum(boot_means <= 0.0)) / (n_bootstrap + 1.0))
            if p_val < worst_p:
                worst_p = p_val
                worst_model = model
                worst_diff = obs

        if worst_model is None or worst_p >= alpha:
            break

        elimination_rows.append(
            {
                "step": step,
                "removed_model": worst_model,
                "reference_best_model": best,
                "bootstrap_p_value": worst_p,
                "mean_loss_diff_removed_minus_best": worst_diff,
            }
        )
        active.remove(worst_model)
        step += 1

    summary_rows = []
    for model in aligned.columns:
        summary_rows.append(
            {
                "model": model,
                "in_mcs": bool(model in active),
                "mean_loss": float(aligned[model].mean()),
            }
        )

    mcs_df = pd.DataFrame(summary_rows).sort_values(["in_mcs", "mean_loss"], ascending=[False, True]).reset_index(
        drop=True
    )
    elim_df = pd.DataFrame(elimination_rows)
    return mcs_df, elim_df
