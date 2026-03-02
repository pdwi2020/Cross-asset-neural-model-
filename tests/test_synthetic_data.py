import numpy as np

from cross_asset_research.config import DataConfig
from cross_asset_research.synthetic_data import generate_synthetic_cross_asset_data


def test_generate_synthetic_data_schema_and_ranges() -> None:
    cfg = DataConfig(assets=["btc", "eurusd", "spx"], n_days=180, start_date="2020-01-01")
    bundle = generate_synthetic_cross_asset_data(cfg, seed=7)
    df = bundle.frame

    assert len(df) == 180
    assert set(np.unique(df["regime_id"]).tolist()).issubset({0, 1, 2})

    for asset in cfg.assets:
        assert f"{asset}_ret" in df.columns
        assert f"{asset}_rv" in df.columns
        assert f"{asset}_jump_count" in df.columns
        assert f"log_{asset}_rv" in df.columns

        assert (df[f"{asset}_rv"] >= 0).all()
        assert (df[f"{asset}_jump_count"].isin([0, 1])).all()
