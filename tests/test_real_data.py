from pathlib import Path

import numpy as np
import pandas as pd

from cross_asset_research.real_data import build_cross_asset_daily_frame


def _write_intraday_csv(path: Path, *, seed: int, start: str = "2024-01-01", days: int = 6) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=days * 24 * 60, freq="min", tz="UTC")
    log_ret = rng.normal(0.0, 0.0008, len(idx))
    log_price = np.cumsum(log_ret)
    price = 100.0 * np.exp(log_price)

    df = pd.DataFrame({"timestamp": idx.astype(str), "close": price})
    df.to_csv(path, index=False)


def test_build_cross_asset_daily_frame_from_intraday_csvs(tmp_path: Path) -> None:
    btc = tmp_path / "btc.csv"
    spx = tmp_path / "spx.csv"
    _write_intraday_csv(btc, seed=1)
    _write_intraday_csv(spx, seed=2)

    out = build_cross_asset_daily_frame(
        {"btc": str(btc), "spx": str(spx)},
        timezone="UTC",
        jump_z=4.0,
        min_obs_per_day=60,
    )

    assert not out.empty
    assert {"btc_ret", "btc_rv", "btc_jump_count", "log_btc_rv"}.issubset(out.columns)
    assert {"spx_ret", "spx_rv", "spx_jump_count", "log_spx_rv"}.issubset(out.columns)
    assert {"regime_id", "regime_label"}.issubset(out.columns)
    assert set(out["regime_id"].unique()).issubset({0, 1, 2})
