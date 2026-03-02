import numpy as np

from cross_asset_research.baselines import StudentTHARModel
from cross_asset_research.config import DataConfig, FeatureConfig
from cross_asset_research.evaluation import evaluate_model
from cross_asset_research.features import build_cross_asset_features, get_target_columns
from cross_asset_research.risk_backtests import run_var_es_backtest
from cross_asset_research.synthetic_data import generate_synthetic_cross_asset_data


def test_student_t_har_forecast_and_eval() -> None:
    assets = ["btc", "eurusd", "spx"]
    raw = generate_synthetic_cross_asset_data(DataConfig(assets=assets, n_days=500), seed=17).frame
    feat = build_cross_asset_features(raw, assets=assets, config=FeatureConfig())

    train = feat.iloc[:320].copy()
    test = feat.iloc[320:420].copy()

    model = StudentTHARModel()
    model.fit(train, assets)
    pred = model.predict(test)

    assert pred.distribution == "student_t"
    assert pred.dof is not None
    assert (pred.dof.to_numpy(dtype=float) > 4.0).all()

    y_true = test[get_target_columns(assets)].copy()
    y_true.columns = assets

    ev = evaluate_model(
        model_name=model.name,
        y_true=y_true,
        mu_pred=pred.mu,
        sigma_pred=pred.sigma,
        distribution=pred.distribution,
        dof_pred=pred.dof,
        assets=assets,
    )
    assert np.isfinite(ev.aggregate_nll)

    risk = run_var_es_backtest(
        model_name=model.name,
        y_true=y_true,
        mu_pred=pred.mu,
        sigma_pred=pred.sigma,
        distribution=pred.distribution,
        dof_pred=pred.dof,
        assets=assets,
        alpha=0.95,
    )
    assert len(risk) == len(assets)
    assert {"observed_exceed_rate", "avg_es_level"}.issubset(risk.columns)
