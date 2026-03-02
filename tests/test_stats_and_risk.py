import numpy as np
import pandas as pd

from cross_asset_research.risk_backtests import run_var_es_backtest
from cross_asset_research.stats_tests import (
    diebold_mariano_test,
    model_confidence_set,
    pairwise_dm_vs_baseline,
    spa_test,
)


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


def test_spa_and_mcs_outputs() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(4)
    losses = {
        "naive_last_surface": pd.Series(rng.normal(1.0, 0.1, len(idx)), index=idx),
        "har_rv": pd.Series(rng.normal(0.9, 0.1, len(idx)), index=idx),
        "garch11_qml": pd.Series(rng.normal(0.95, 0.1, len(idx)), index=idx),
    }

    spa = spa_test(losses, benchmark="naive_last_surface", n_bootstrap=60, block_size=5, seed=0)
    assert not spa.empty
    assert {"benchmark", "model", "spa_global_p", "model_one_sided_p"}.issubset(spa.columns)

    mcs, elim = model_confidence_set(losses, alpha=0.1, n_bootstrap=60, block_size=5, seed=0)
    assert not mcs.empty
    assert {"model", "in_mcs", "mean_loss"}.issubset(mcs.columns)
    assert isinstance(elim, pd.DataFrame)
