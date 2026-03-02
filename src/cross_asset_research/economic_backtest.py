"""Economic backtests for volatility-managed strategies with transaction costs."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _max_drawdown(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def _strategy_weights(
    mu_log_rv: pd.Series,
    *,
    target_daily_vol: float,
    max_leverage: float,
) -> pd.Series:
    pred_rv = np.maximum(np.expm1(mu_log_rv.to_numpy(dtype=float)), 1e-10)
    pred_vol = np.sqrt(pred_rv)
    w = target_daily_vol / np.maximum(pred_vol, 1e-6)
    w = np.clip(w, 0.0, max_leverage)
    return pd.Series(w, index=mu_log_rv.index)


def _performance_row(
    *,
    model: str,
    asset: str,
    gross: pd.Series,
    net: pd.Series,
    turnover: pd.Series,
    annualization: int,
    transaction_cost_bps: float,
) -> dict:
    mu_g = float(gross.mean())
    sd_g = float(gross.std(ddof=1)) if len(gross) > 1 else 0.0
    mu_n = float(net.mean())
    sd_n = float(net.std(ddof=1)) if len(net) > 1 else 0.0

    ann_ret_gross = mu_g * annualization
    ann_vol_gross = sd_g * np.sqrt(annualization)
    ann_ret_net = mu_n * annualization
    ann_vol_net = sd_n * np.sqrt(annualization)

    downside = net[net < 0]
    downside_vol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    ann_downside = downside_vol * np.sqrt(annualization)

    mdd = _max_drawdown(net)

    return {
        "model": model,
        "asset": asset,
        "n_obs": int(len(net)),
        "ann_return_gross": ann_ret_gross,
        "ann_vol_gross": ann_vol_gross,
        "sharpe_gross": float(ann_ret_gross / ann_vol_gross) if ann_vol_gross > 1e-12 else 0.0,
        "ann_return_net": ann_ret_net,
        "ann_vol_net": ann_vol_net,
        "sharpe_net": float(ann_ret_net / ann_vol_net) if ann_vol_net > 1e-12 else 0.0,
        "sortino_net": float(ann_ret_net / ann_downside) if ann_downside > 1e-12 else 0.0,
        "max_drawdown_net": mdd,
        "calmar_net": float(ann_ret_net / abs(mdd)) if np.isfinite(mdd) and abs(mdd) > 1e-12 else 0.0,
        "hit_rate_net": float((net > 0).mean()) if len(net) > 0 else 0.0,
        "avg_turnover": float(turnover.mean()) if len(turnover) > 0 else 0.0,
        "annual_cost_drag_bps": float(turnover.mean() * (transaction_cost_bps / 1e4) * annualization * 1e4)
        if len(turnover) > 0
        else 0.0,
    }


def run_economic_backtests(
    *,
    predictions_by_model: Dict[str, Dict[str, pd.DataFrame]],
    raw_df: pd.DataFrame,
    assets: Iterable[str],
    annualization: int = 252,
    target_daily_vol: float = 0.01,
    max_leverage: float = 3.0,
    transaction_cost_bps: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run volatility-managed economic backtests across models and assets."""

    assets = list(assets)
    tc = transaction_cost_bps / 1e4

    rows = []
    model_portfolio_returns: Dict[str, pd.Series] = {}

    for model, payload in predictions_by_model.items():
        mu = payload["mu"].copy()
        asset_nets: Dict[str, pd.Series] = {}

        for asset in assets:
            if asset not in mu.columns:
                continue
            ret_col = f"{asset}_ret"
            if ret_col not in raw_df.columns:
                continue

            aligned = pd.concat(
                [mu[asset].rename("mu"), raw_df[ret_col].rename("ret")], axis=1, join="inner"
            ).dropna()
            if len(aligned) < 5:
                continue

            w = _strategy_weights(
                aligned["mu"], target_daily_vol=target_daily_vol, max_leverage=max_leverage
            )
            turnover = (w - w.shift(1).fillna(0.0)).abs()

            gross = w * aligned["ret"]
            net = gross - turnover * tc

            rows.append(
                _performance_row(
                    model=model,
                    asset=asset,
                    gross=gross,
                    net=net,
                    turnover=turnover,
                    annualization=annualization,
                    transaction_cost_bps=transaction_cost_bps,
                )
            )
            asset_nets[asset] = net

        if asset_nets:
            port = pd.concat(asset_nets.values(), axis=1).mean(axis=1)
            w_proxy = pd.Series(0.0, index=port.index)
            rows.append(
                _performance_row(
                    model=model,
                    asset="portfolio_equal_weight",
                    gross=port,
                    net=port,
                    turnover=w_proxy,
                    annualization=annualization,
                    transaction_cost_bps=transaction_cost_bps,
                )
            )
            model_portfolio_returns[model] = port

    econ_df = pd.DataFrame(rows)
    if not econ_df.empty:
        order = econ_df[econ_df["asset"] == "portfolio_equal_weight"].sort_values(
            "sharpe_net", ascending=False
        )["model"]
        if len(order) > 0:
            econ_df["model"] = pd.Categorical(econ_df["model"], categories=order.tolist(), ordered=True)
            econ_df = econ_df.sort_values(["asset", "model"]).reset_index(drop=True)

    pnl_df = pd.DataFrame(model_portfolio_returns).sort_index() if model_portfolio_returns else pd.DataFrame()
    return econ_df, pnl_df
