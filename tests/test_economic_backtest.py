import pandas as pd

from cross_asset_research.baselines import HARModel
from cross_asset_research.economic_backtest import run_economic_backtests
from cross_asset_research.config import DataConfig, FeatureConfig
from cross_asset_research.features import build_cross_asset_features, get_target_columns
from cross_asset_research.synthetic_data import generate_synthetic_cross_asset_data


def test_economic_backtest_outputs() -> None:
    assets = ["btc", "eurusd", "spx"]
    raw = generate_synthetic_cross_asset_data(DataConfig(assets=assets, n_days=360), seed=5).frame
    feat = build_cross_asset_features(raw, assets=assets, config=FeatureConfig())

    train = feat.iloc[:220].copy()
    test = feat.iloc[220:300].copy()

    model = HARModel()
    model.fit(train, assets)
    pred = model.predict(test)

    y = test[get_target_columns(assets)].copy()
    y.columns = assets

    predictions_by_model = {
        model.name: {
            "y_true": y,
            "mu": pred.mu,
            "sigma": pred.sigma,
            "dof": None,
            "distribution": "gaussian",
        }
    }

    econ_df, pnl_df = run_economic_backtests(
        predictions_by_model=predictions_by_model,
        raw_df=raw,
        assets=assets,
        annualization=252,
        target_daily_vol=0.01,
        max_leverage=3.0,
        transaction_cost_bps=5.0,
    )

    assert not econ_df.empty
    assert {"model", "asset", "sharpe_net", "ann_return_net", "max_drawdown_net"}.issubset(econ_df.columns)
    assert isinstance(pnl_df, pd.DataFrame)
