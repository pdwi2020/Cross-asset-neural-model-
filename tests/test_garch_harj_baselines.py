from cross_asset_research.baselines import GARCH11Model, HARJModel, default_models
from cross_asset_research.config import DataConfig, FeatureConfig
from cross_asset_research.features import build_cross_asset_features
from cross_asset_research.synthetic_data import generate_synthetic_cross_asset_data


def test_default_models_include_new_baselines() -> None:
    names = [m.name for m in default_models(include_student_t=True, include_garch=True, include_har_j=True)]
    assert "har_j_rv" in names
    assert "garch11_qml" in names


def test_garch_and_harj_predict_shapes() -> None:
    assets = ["btc", "eurusd", "spx"]
    raw = generate_synthetic_cross_asset_data(DataConfig(assets=assets, n_days=450), seed=12).frame
    feat = build_cross_asset_features(raw, assets=assets, config=FeatureConfig())

    train = feat.iloc[:280].copy()
    test = feat.iloc[280:360].copy()

    harj = HARJModel()
    harj.fit(train, assets)
    pred_harj = harj.predict(test)
    assert list(pred_harj.mu.columns) == assets

    garch = GARCH11Model()
    garch.fit(train, assets)
    pred_garch = garch.predict(test)
    assert list(pred_garch.mu.columns) == assets
    assert (pred_garch.sigma.to_numpy() > 0).all()
