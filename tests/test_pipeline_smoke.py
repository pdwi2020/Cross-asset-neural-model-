from pathlib import Path

import numpy as np
import pandas as pd

from cross_asset_research.config import PipelineConfig
from cross_asset_research.pipeline import run_doctoral_pipeline


def test_pipeline_smoke_generates_artifacts(tmp_path: Path) -> None:
    cfg = PipelineConfig(seed=123, quick=True, include_lstm=False)
    cfg.reporting.output_dir = str(tmp_path)
    cfg.reporting.run_name = "smoke"

    out = run_doctoral_pipeline(cfg)

    assert out["leaderboard"].shape[0] >= 3
    assert out["manifest_path"].exists()
    assert out["summary_markdown"].exists()

    fig_dir = tmp_path / "smoke" / "figures"
    table_dir = tmp_path / "smoke" / "tables"
    assert fig_dir.exists()
    assert table_dir.exists()
    assert len(list(fig_dir.glob("*.png"))) >= 8


def _write_intraday(path: Path, *, seed: int, days: int = 14) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 24 * 60, freq="min", tz="UTC")
    ret = rng.normal(0.0, 0.0007, len(idx))
    price = 100.0 * np.exp(np.cumsum(ret))
    pd.DataFrame({"timestamp": idx.astype(str), "close": price}).to_csv(path, index=False)


def test_pipeline_real_data_smoke(tmp_path: Path) -> None:
    in_dir = tmp_path / "intraday"
    in_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "btc": in_dir / "btc.csv",
        "eurusd": in_dir / "eurusd.csv",
        "spx": in_dir / "spx.csv",
    }
    for i, (_, p) in enumerate(files.items(), start=1):
        _write_intraday(p, seed=i, days=14)

    cfg = PipelineConfig(seed=321, quick=True, include_lstm=False)
    cfg.data.data_source = "real"
    cfg.data.assets = list(files.keys())
    cfg.data.intraday_file_map = {k: str(v) for k, v in files.items()}
    cfg.reporting.output_dir = str(tmp_path)
    cfg.reporting.run_name = "real_smoke"

    out = run_doctoral_pipeline(cfg)
    assert out["leaderboard"].shape[0] >= 3
    assert (tmp_path / "real_smoke" / "manifest.json").exists()
