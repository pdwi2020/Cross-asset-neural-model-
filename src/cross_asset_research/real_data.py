"""Real-data ingestion: intraday bars to daily cross-asset research features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _infer_column(columns: list[str], candidates: list[str]) -> str:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise ValueError(f"Could not infer required column from candidates={candidates}")


def load_intraday_csv(
    path: str | Path,
    *,
    timestamp_col: str | None = None,
    price_col: str | None = None,
    timezone: str = "UTC",
    jump_z: float = 4.0,
) -> pd.DataFrame:
    """Load one intraday CSV and compute minute-level returns/jump indicators."""

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")

    ts_col = timestamp_col or _infer_column(
        list(df.columns),
        ["timestamp", "datetime", "date", "time", "Date", "Datetime", "Timestamp"],
    )
    px_col = price_col or _infer_column(
        list(df.columns),
        ["close", "adj_close", "adj close", "price", "Close", "Adj Close", "Price"],
    )

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    valid = ts.notna()
    if valid.sum() == 0:
        raise ValueError(f"Failed to parse datetime column {ts_col!r} in {path}")

    out = pd.DataFrame(index=ts[valid])
    out.index = out.index.tz_convert(timezone)
    out["price"] = pd.to_numeric(df.loc[valid, px_col], errors="coerce").to_numpy()
    out = out.dropna(subset=["price"])
    out = out[out["price"] > 0]
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if len(out) < 5:
        raise ValueError(f"Insufficient valid observations in {path}")

    out["log_ret"] = np.log(out["price"]).diff().fillna(0.0)
    rolling_std = out["log_ret"].rolling(60, min_periods=30).std()
    threshold = jump_z * rolling_std
    out["jump_flag"] = (out["log_ret"].abs() > threshold).astype(float).fillna(0.0)
    out["rv_component"] = out["log_ret"] ** 2
    return out


def intraday_to_daily_features(
    intraday_df: pd.DataFrame,
    *,
    min_obs_per_day: int = 30,
) -> pd.DataFrame:
    """Aggregate intraday features into daily realized features."""

    day = intraday_df.index.floor("D")
    grp = intraday_df.groupby(day)

    daily = pd.DataFrame(index=sorted(grp.groups.keys()))
    daily["obs_count"] = grp["log_ret"].count()
    daily["daily_ret"] = grp["log_ret"].sum()
    daily["daily_rv"] = grp["rv_component"].sum()
    daily["daily_jump_count"] = grp["jump_flag"].sum()

    daily = daily[daily["obs_count"] >= min_obs_per_day].copy()
    daily["log_daily_rv"] = np.log1p(daily["daily_rv"].clip(lower=0.0))
    return daily


def build_cross_asset_daily_frame(
    file_map: Dict[str, str],
    *,
    timezone: str = "UTC",
    jump_z: float = 4.0,
    min_obs_per_day: int = 30,
) -> pd.DataFrame:
    """Build daily cross-asset frame from per-asset intraday CSV paths."""

    if not file_map:
        raise ValueError("file_map cannot be empty for real-data mode")

    merged: pd.DataFrame | None = None
    for asset, path in file_map.items():
        intra = load_intraday_csv(path, timezone=timezone, jump_z=jump_z)
        daily = intraday_to_daily_features(intra, min_obs_per_day=min_obs_per_day)

        part = pd.DataFrame(index=daily.index)
        part[f"{asset}_ret"] = daily["daily_ret"]
        part[f"{asset}_rv"] = daily["daily_rv"]
        part[f"{asset}_jump_count"] = daily["daily_jump_count"]
        part[f"log_{asset}_rv"] = daily["log_daily_rv"]

        merged = part if merged is None else merged.join(part, how="inner")

    if merged is None or merged.empty:
        raise ValueError("No overlapping daily rows found across assets")

    log_cols = [c for c in merged.columns if c.startswith("log_") and c.endswith("_rv")]
    mean_log_rv = merged[log_cols].mean(axis=1)
    q1, q2 = mean_log_rv.quantile([0.33, 0.67]).tolist()
    regime_id = np.where(mean_log_rv <= q1, 0, np.where(mean_log_rv <= q2, 1, 2))

    merged = merged.sort_index()
    merged["regime_id"] = regime_id.astype(int)
    labels = {0: "low_vol", 1: "mid_vol", 2: "stress"}
    merged["regime_label"] = merged["regime_id"].map(labels)
    return merged
