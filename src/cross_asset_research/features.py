"""Cross-asset feature engineering utilities."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List

import numpy as np
import pandas as pd

from .config import FeatureConfig


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.replace(0.0, np.nan)
    return a / denom


def build_cross_asset_features(df: pd.DataFrame, assets: Iterable[str], config: FeatureConfig) -> pd.DataFrame:
    """Build lagged volatility, contagion, correlation, and beta features."""

    assets = list(assets)
    out = df.copy()

    for asset in assets:
        log_rv = f"log_{asset}_rv"
        if log_rv not in out.columns:
            out[log_rv] = np.log1p(out[f"{asset}_rv"]) if f"{asset}_rv" in out.columns else np.nan

        out[f"{asset}_log_rv_lag1"] = out[log_rv].shift(1)
        out[f"{asset}_log_rv_wavg"] = out[log_rv].rolling(config.weekly_lag).mean().shift(1)
        out[f"{asset}_log_rv_mavg"] = out[log_rv].rolling(config.monthly_lag).mean().shift(1)
        out[f"{asset}_jump_lag1"] = out[f"{asset}_jump_count"].shift(1)

    # Cross-asset rolling correlations.
    for left, right in combinations(assets, 2):
        c = out[f"{left}_ret"].rolling(config.corr_window).corr(out[f"{right}_ret"])
        out[f"corr_{left}_{right}_{config.corr_window}d"] = c

    # Rolling beta of each asset vs benchmark (last asset by default).
    benchmark = assets[-1]
    bench_ret = out[f"{benchmark}_ret"]
    bench_var = bench_ret.rolling(config.beta_window).var()
    for asset in assets:
        if asset == benchmark:
            continue
        cov = out[f"{asset}_ret"].rolling(config.beta_window).cov(bench_ret)
        out[f"beta_{asset}_vs_{benchmark}_{config.beta_window}d"] = _safe_div(cov, bench_var)

    # Volatility spillover lags from other assets.
    for asset in assets:
        others: List[str] = [a for a in assets if a != asset]
        for other in others:
            out[f"{asset}_spill_{other}_lag1"] = out[f"log_{other}_rv"].shift(1)

    return out.dropna().copy()


def get_feature_columns(df: pd.DataFrame, assets: Iterable[str]) -> List[str]:
    """Select model feature columns from engineered frame."""

    assets = list(assets)
    cols: List[str] = []
    for asset in assets:
        cols.extend(
            [
                f"{asset}_log_rv_lag1",
                f"{asset}_log_rv_wavg",
                f"{asset}_log_rv_mavg",
                f"{asset}_jump_lag1",
            ]
        )

    corr_cols = [c for c in df.columns if c.startswith("corr_")]
    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    spill_cols = [c for c in df.columns if "_spill_" in c]
    cols.extend(sorted(corr_cols))
    cols.extend(sorted(beta_cols))
    cols.extend(sorted(spill_cols))

    cols = [c for c in cols if c in df.columns]
    return cols


def get_target_columns(assets: Iterable[str]) -> List[str]:
    """Target columns for next-day log realized volatility."""

    return [f"log_{asset}_rv" for asset in assets]
