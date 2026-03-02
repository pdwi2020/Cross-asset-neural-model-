import numpy as np
import pandas as pd

from cross_asset_research.risk_backtests import run_var_es_backtest
from cross_asset_research.stats_tests import diebold_mariano_test, pairwise_dm_vs_baseline


def test_dm_detects_better_model() -> None:
    rng = np.random.default_rng(0)
    better = rng.normal(0.8, 0.1, size=300)
    worse = rng.normal(1.0, 0.1, size=300)

    dm = diebold_mariano_test(worse, better)
    assert dm.p_value < 0.05
    assert dm.mean_loss_diff > 0


def test_pairwise_dm_vs_baseline_schema() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    losses = {
        "base": pd.Series(np.linspace(0.7, 1.1, len(idx)), index=idx),
        "challenger": pd.Series(np.linspace(0.6, 0.9, len(idx)), index=idx),
    }
    out = pairwise_dm_vs_baseline(losses, baseline="base")
    assert not out.empty
    assert {"model", "baseline", "p_value", "holm_reject_h0", "bh_reject_h0"}.issubset(out.columns)


def test_var_backtest_outputs_expected_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    assets = ["btc", "spx"]

    y = pd.DataFrame({"btc": np.random.normal(0.2, 0.03, 100), "spx": np.random.normal(0.15, 0.02, 100)}, index=idx)
    mu = y - 0.01
    sigma = pd.DataFrame({"btc": 0.02, "spx": 0.015}, index=idx)

    out = run_var_es_backtest(
        model_name="x",
        y_true=y,
        mu_pred=mu,
        sigma_pred=sigma,
        assets=assets,
        alpha=0.95,
    )

    assert len(out) == 2
    assert {"model", "asset", "observed_exceed_rate", "kupiec_p_value", "christoffersen_p_value"}.issubset(
        out.columns
    )
