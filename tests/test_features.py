from cross_asset_research.config import DataConfig, FeatureConfig
from cross_asset_research.features import build_cross_asset_features, get_feature_columns, get_target_columns
from cross_asset_research.synthetic_data import generate_synthetic_cross_asset_data


def test_feature_engineering_outputs_expected_columns() -> None:
    assets = ["btc", "eurusd", "spx"]
    raw = generate_synthetic_cross_asset_data(DataConfig(assets=assets, n_days=160), seed=11).frame
    feat = build_cross_asset_features(raw, assets=assets, config=FeatureConfig())

    assert not feat.empty
    assert feat.isna().sum().sum() == 0

    feature_cols = get_feature_columns(feat, assets)
    target_cols = get_target_columns(assets)

    assert len(feature_cols) > 10
    for c in target_cols:
        assert c in feat.columns
