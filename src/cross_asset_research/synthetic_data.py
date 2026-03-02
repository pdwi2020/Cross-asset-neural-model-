"""Synthetic cross-asset data generation with regime shifts and contagion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class SyntheticDataBundle:
    """Container for generated daily cross-asset series."""

    frame: pd.DataFrame
    regime_labels: Dict[int, str]


def _sample_regimes(transition: np.ndarray, n_days: int, rng: np.random.Generator) -> np.ndarray:
    states = np.zeros(n_days, dtype=int)
    states[0] = 0
    for t in range(1, n_days):
        states[t] = rng.choice(np.arange(transition.shape[0]), p=transition[states[t - 1]])
    return states


def _regime_params(n_assets: int) -> Dict[int, Dict[str, np.ndarray]]:
    low_vol = {
        "vol": np.full(n_assets, 0.012),
        "jump_intensity": np.full(n_assets, 0.01),
        "jump_scale": np.full(n_assets, 0.03),
        "corr": np.array(
            [
                [1.0, 0.12, 0.06],
                [0.12, 1.0, 0.18],
                [0.06, 0.18, 1.0],
            ]
        )[:n_assets, :n_assets],
    }
    mid_vol = {
        "vol": np.full(n_assets, 0.020),
        "jump_intensity": np.full(n_assets, 0.03),
        "jump_scale": np.full(n_assets, 0.05),
        "corr": np.array(
            [
                [1.0, 0.22, 0.16],
                [0.22, 1.0, 0.28],
                [0.16, 0.28, 1.0],
            ]
        )[:n_assets, :n_assets],
    }
    stress = {
        "vol": np.full(n_assets, 0.035),
        "jump_intensity": np.full(n_assets, 0.08),
        "jump_scale": np.full(n_assets, 0.09),
        "corr": np.array(
            [
                [1.0, 0.42, 0.37],
                [0.42, 1.0, 0.45],
                [0.37, 0.45, 1.0],
            ]
        )[:n_assets, :n_assets],
    }
    return {0: low_vol, 1: mid_vol, 2: stress}


def generate_synthetic_cross_asset_data(config: DataConfig, *, seed: int = 42) -> SyntheticDataBundle:
    """Generate cross-asset daily returns, realized variance and jump counts.

    The simulation uses regime-dependent correlation, volatility, and jump intensity,
    with mild volatility spillovers across assets.
    """

    rng = np.random.default_rng(seed)
    assets: List[str] = list(config.assets)
    n_assets = len(assets)

    transition = np.asarray(config.regime_transition, dtype=float)
    if transition.shape != (3, 3):
        raise ValueError("regime_transition must be 3x3")

    regimes = _sample_regimes(transition, config.n_days, rng)
    regime_map = {0: "low_vol", 1: "mid_vol", 2: "stress"}
    params = _regime_params(n_assets)

    ret = np.zeros((config.n_days, n_assets), dtype=float)
    rv = np.zeros((config.n_days, n_assets), dtype=float)
    jumps = np.zeros((config.n_days, n_assets), dtype=float)

    prev_rv = np.full(n_assets, 1e-4, dtype=float)

    for t in range(config.n_days):
        state = int(regimes[t])
        p = params[state]
        base_vol = p["vol"]

        # Mild contagion from previous realized variance across assets.
        spillover = 0.15 * (prev_rv - np.mean(prev_rv))
        vol_t = np.clip(base_vol + spillover, 0.004, 0.12)

        cov = np.outer(vol_t, vol_t) * p["corr"]
        eps = rng.multivariate_normal(mean=np.zeros(n_assets), cov=cov)

        jump_indicator = rng.binomial(1, p["jump_intensity"], size=n_assets)
        jump_sizes = jump_indicator * rng.normal(loc=0.0, scale=p["jump_scale"], size=n_assets)

        ret_t = eps + jump_sizes
        rv_t = np.square(ret_t) + rng.uniform(0.0, 0.00005, size=n_assets)

        ret[t] = ret_t
        rv[t] = np.clip(rv_t, 1e-10, None)
        jumps[t] = jump_indicator
        prev_rv = rv[t]

    idx = pd.date_range(config.start_date, periods=config.n_days, freq="D")
    df = pd.DataFrame(index=idx)
    df["regime_id"] = regimes
    df["regime_label"] = [regime_map[int(x)] for x in regimes]

    for i, asset in enumerate(assets):
        df[f"{asset}_ret"] = ret[:, i]
        df[f"{asset}_rv"] = rv[:, i]
        df[f"{asset}_jump_count"] = jumps[:, i]
        df[f"log_{asset}_rv"] = np.log1p(df[f"{asset}_rv"])

    return SyntheticDataBundle(frame=df, regime_labels=regime_map)
