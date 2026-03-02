"""End-to-end doctoral pipeline for cross-asset volatility research."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .baselines import default_models, fit_predict_models
from .config import PipelineConfig
from .deep_models import probabilistic_lstm_forecast
from .evaluation import ModelEvaluation, evaluate_model, evaluations_to_frame
from .features import build_cross_asset_features, get_feature_columns, get_target_columns
from .real_data import build_cross_asset_daily_frame
from .reporting import generate_advanced_figures, prepare_report_dirs, save_tables, write_summary_markdown
from .risk_backtests import run_var_es_backtest
from .stats_tests import pairwise_dm_vs_baseline
from .synthetic_data import generate_synthetic_cross_asset_data
from .walkforward import generate_walkforward_splits, summarize_splits


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _concat_non_overlapping(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).mean(numeric_only=True)
    return out


def _asset_metrics_frame(evals: Iterable[ModelEvaluation]) -> pd.DataFrame:
    rows: List[dict] = []
    for ev in evals:
        for m in ev.by_asset:
            row = m.__dict__.copy()
            row["model"] = ev.model
            rows.append(row)
    return pd.DataFrame(rows)


def _adaptive_walkforward_params(n_samples: int) -> Dict[str, int] | None:
    """Derive conservative split sizes for shorter real-data histories."""

    if n_samples < 6:
        return None

    train = max(int(0.6 * n_samples), 4)
    val = max(int(0.2 * n_samples), 1)
    test = max(int(0.1 * n_samples), 1)

    while train + val + test > n_samples and train > 4:
        train -= 1
    while train + val + test > n_samples and val > 1:
        val -= 1
    while train + val + test > n_samples and test > 1:
        test -= 1

    if train + val + test > n_samples:
        return None
    return {
        "train_size": train,
        "val_size": val,
        "test_size": test,
        "step_size": max(test // 2, 1),
    }


def _append_prediction_segment(
    store: Dict[str, Dict[str, List[pd.DataFrame]]],
    *,
    model_name: str,
    y_true: pd.DataFrame,
    mu: pd.DataFrame,
    sigma: pd.DataFrame,
    distribution: str = "gaussian",
    dof: pd.DataFrame | None = None,
) -> None:
    if model_name not in store:
        store[model_name] = {"y_true": [], "mu": [], "sigma": [], "dof": [], "distribution": distribution}
    store[model_name]["y_true"].append(y_true)
    store[model_name]["mu"].append(mu)
    store[model_name]["sigma"].append(sigma)
    if dof is not None:
        store[model_name]["dof"].append(dof)


def run_doctoral_pipeline(config: PipelineConfig | None = None) -> Dict[str, object]:
    """Run full walk-forward modeling, testing, and reporting workflow."""

    cfg = config or PipelineConfig()
    cfg.apply_quick_mode()
    _set_global_seed(cfg.seed)

    assets = list(cfg.data.assets)
    if cfg.data.data_source == "real":
        raw_df = build_cross_asset_daily_frame(
            cfg.data.intraday_file_map,
            timezone=cfg.data.timezone,
            jump_z=cfg.data.jump_z,
            min_obs_per_day=cfg.data.min_obs_per_day,
        )
    else:
        data_bundle = generate_synthetic_cross_asset_data(cfg.data, seed=cfg.seed)
        raw_df = data_bundle.frame.copy()
    engineered = build_cross_asset_features(raw_df, assets=assets, config=cfg.features)
    if engineered.empty and len(raw_df) >= 12:
        cfg.features.corr_window = min(cfg.features.corr_window, max(5, len(raw_df) // 6))
        cfg.features.beta_window = min(cfg.features.beta_window, max(5, len(raw_df) // 6))
        cfg.features.weekly_lag = min(cfg.features.weekly_lag, max(2, len(raw_df) // 20))
        cfg.features.monthly_lag = min(cfg.features.monthly_lag, max(5, len(raw_df) // 8))
        engineered = build_cross_asset_features(raw_df, assets=assets, config=cfg.features)
    if engineered.empty:
        raise ValueError(
            "Feature engineering produced zero rows. Increase data length or reduce feature windows."
        )

    feature_cols = get_feature_columns(engineered, assets)
    target_cols = get_target_columns(assets)

    splits = generate_walkforward_splits(
        len(engineered),
        train_size=cfg.walkforward.train_size,
        val_size=cfg.walkforward.val_size,
        test_size=cfg.walkforward.test_size,
        step_size=cfg.walkforward.step_size,
    )
    if not splits:
        adaptive = _adaptive_walkforward_params(len(engineered))
        if adaptive is not None:
            splits = generate_walkforward_splits(len(engineered), **adaptive)
            if splits:
                cfg.walkforward.train_size = adaptive["train_size"]
                cfg.walkforward.val_size = adaptive["val_size"]
                cfg.walkforward.test_size = adaptive["test_size"]
                cfg.walkforward.step_size = adaptive["step_size"]
        if not splits:
            raise ValueError(
                "No walk-forward splits generated. Increase data length or reduce window sizes. "
                f"n_samples={len(engineered)}."
            )

    split_boundaries_df = pd.DataFrame(summarize_splits(splits))
    split_perf_rows: List[dict] = []
    pred_segments: Dict[str, Dict[str, List[pd.DataFrame]]] = {}

    for split_id, split in enumerate(splits):
        train_df = engineered.iloc[split.train_idx].copy()
        val_df = engineered.iloc[split.val_idx].copy()
        test_df = engineered.iloc[split.test_idx].copy()
        train_val_df = pd.concat([train_df, val_df], axis=0)

        forecasts = fit_predict_models(
            default_models(include_student_t=cfg.include_student_t_baseline),
            train_df=train_val_df,
            test_df=test_df,
            assets=assets,
        )

        if cfg.include_lstm:
            lstm_forecast, lstm_diag = probabilistic_lstm_forecast(
                train_df=train_val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                assets=assets,
                cfg=cfg.model,
                seed=cfg.seed + split_id,
            )
            forecasts["prob_lstm_gaussian"] = lstm_forecast
            if lstm_diag is not None:
                split_perf_rows.append(
                    {
                        "split_id": split_id,
                        "model": "prob_lstm_gaussian",
                        "epochs_run": lstm_diag.epochs_run,
                        "best_val_nll": lstm_diag.best_val_nll,
                    }
                )

        y_true_split = test_df[target_cols].copy()
        y_true_split.columns = assets

        for model_name, forecast in forecasts.items():
            mu = forecast.mu[assets].copy()
            sigma = forecast.sigma[assets].copy()
            y_true = y_true_split.loc[mu.index].copy()
            distribution = getattr(forecast, "distribution", "gaussian")
            dof = getattr(forecast, "dof", None)
            if dof is not None:
                dof = dof[assets].copy()

            _append_prediction_segment(
                pred_segments,
                model_name=model_name,
                y_true=y_true,
                mu=mu,
                sigma=sigma,
                distribution=distribution,
                dof=dof,
            )

            ev_split = evaluate_model(
                model_name=model_name,
                y_true=y_true,
                mu_pred=mu,
                sigma_pred=sigma,
                distribution=distribution,
                dof_pred=dof,
                assets=assets,
            )
            split_perf_rows.append(
                {
                    "split_id": split_id,
                    "model": model_name,
                    "aggregate_rmse": ev_split.aggregate_rmse,
                    "aggregate_mae": ev_split.aggregate_mae,
                    "aggregate_nll": ev_split.aggregate_nll,
                    "aggregate_coverage_95": ev_split.aggregate_coverage_95,
                }
            )

    predictions_by_model: Dict[str, Dict[str, pd.DataFrame]] = {}
    for model_name, payload in pred_segments.items():
        dof_df = _concat_non_overlapping(payload["dof"]) if payload["dof"] else None
        predictions_by_model[model_name] = {
            "y_true": _concat_non_overlapping(payload["y_true"]),
            "mu": _concat_non_overlapping(payload["mu"]),
            "sigma": _concat_non_overlapping(payload["sigma"]),
            "dof": dof_df,
            "distribution": payload.get("distribution", "gaussian"),
        }

    evals: List[ModelEvaluation] = []
    for model_name, payload in predictions_by_model.items():
        ev = evaluate_model(
            model_name=model_name,
            y_true=payload["y_true"],
            mu_pred=payload["mu"],
            sigma_pred=payload["sigma"],
            distribution=str(payload.get("distribution", "gaussian")),
            dof_pred=payload.get("dof"),
            assets=assets,
        )
        evals.append(ev)

    leaderboard_df = evaluations_to_frame(evals)
    if leaderboard_df.empty:
        raise RuntimeError("No model evaluations were produced.")

    asset_metrics_df = _asset_metrics_frame(evals)
    best_model = str(leaderboard_df.iloc[0]["model"])

    losses_by_model: Dict[str, pd.Series] = {}
    for model_name, payload in predictions_by_model.items():
        aligned = pd.concat([payload["y_true"], payload["mu"]], axis=1, join="inner")
        y = aligned.iloc[:, : len(assets)]
        mu = aligned.iloc[:, len(assets) :]
        mu.columns = y.columns
        losses_by_model[model_name] = ((y - mu) ** 2).mean(axis=1)

    dm_df = pairwise_dm_vs_baseline(
        losses_by_model,
        baseline=best_model,
        horizon=cfg.risk.var_horizon_days,
        alpha=cfg.alpha,
    )

    risk_frames: List[pd.DataFrame] = []
    for model_name, payload in predictions_by_model.items():
        risk_frames.append(
            run_var_es_backtest(
                model_name=model_name,
                y_true=payload["y_true"],
                mu_pred=payload["mu"],
                sigma_pred=payload["sigma"],
                distribution=str(payload.get("distribution", "gaussian")),
                dof_pred=payload.get("dof"),
                assets=assets,
                alpha=cfg.risk.var_alpha,
            )
        )
    risk_df = pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame()

    split_perf_df = pd.DataFrame(split_perf_rows)

    report_dirs = prepare_report_dirs(cfg.reporting.output_dir, cfg.reporting.run_name)
    table_paths = save_tables(
        leaderboard_df=leaderboard_df,
        asset_metrics_df=asset_metrics_df,
        dm_df=dm_df,
        risk_df=risk_df,
        split_perf_df=split_perf_df,
        split_boundaries_df=split_boundaries_df,
        out_tables_dir=report_dirs["tables"],
    )

    if cfg.reporting.save_predictions_csv:
        pred_dir = report_dirs["tables"] / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for model_name, payload in predictions_by_model.items():
            out = payload["y_true"].copy()
            out.columns = [f"y_{c}" for c in out.columns]

            mu = payload["mu"].copy()
            mu.columns = [f"mu_{c}" for c in mu.columns]

            sigma = payload["sigma"].copy()
            sigma.columns = [f"sigma_{c}" for c in sigma.columns]

            frames = [out, mu, sigma]
            if payload.get("dof") is not None:
                dof = payload["dof"].copy()
                dof.columns = [f"dof_{c}" for c in dof.columns]
                frames.append(dof)

            merged = pd.concat(frames, axis=1)
            safe_name = model_name.replace("/", "_")
            merged.to_csv(pred_dir / f"{safe_name}_predictions.csv", index=True)

    figure_paths = generate_advanced_figures(
        raw_df=raw_df,
        engineered_df=engineered,
        assets=assets,
        leaderboard_df=leaderboard_df,
        asset_metrics_df=asset_metrics_df,
        dm_df=dm_df,
        risk_df=risk_df,
        split_perf_df=split_perf_df,
        predictions_by_model=predictions_by_model,
        best_model=best_model,
        out_fig_dir=report_dirs["figures"],
        dpi=cfg.reporting.dpi,
    )

    config_summary = {
        "seed": cfg.seed,
        "quick": cfg.quick,
        "data_source": cfg.data.data_source,
        "assets": ",".join(assets),
        "n_days": cfg.data.n_days,
        "walkforward": (
            f"train={cfg.walkforward.train_size},val={cfg.walkforward.val_size},"
            f"test={cfg.walkforward.test_size},step={cfg.walkforward.step_size}"
        ),
        "include_lstm": cfg.include_lstm,
        "include_student_t_baseline": cfg.include_student_t_baseline,
        "var_alpha": cfg.risk.var_alpha,
    }
    summary_md = write_summary_markdown(
        output_root=report_dirs["root"],
        config_summary=config_summary,
        leaderboard_df=leaderboard_df,
        dm_df=dm_df,
        risk_df=risk_df,
        figure_paths=figure_paths,
    )

    manifest = {
        "output_root": str(report_dirs["root"]),
        "best_model": best_model,
        "table_paths": {k: str(v) for k, v in table_paths.items()},
        "figure_paths": {k: str(v) for k, v in figure_paths.items()},
        "summary_markdown": str(summary_md),
        "leaderboard_top": leaderboard_df.head(3).to_dict(orient="records"),
        "num_splits": len(splits),
        "num_models": len(predictions_by_model),
        "config": asdict(cfg),
    }
    manifest_path = report_dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "output_root": report_dirs["root"],
        "summary_markdown": summary_md,
        "leaderboard": leaderboard_df,
        "asset_metrics": asset_metrics_df,
        "dm_tests": dm_df,
        "risk_backtests": risk_df,
        "split_performance": split_perf_df,
        "predictions_by_model": predictions_by_model,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cross-asset doctoral research pipeline")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output artifact root directory")
    parser.add_argument("--run-name", type=str, default="latest", help="Run folder name under output-dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
        help="Input data source mode",
    )
    parser.add_argument("--n-days", type=int, default=None, help="Override number of synthetic days")
    parser.add_argument("--assets", type=str, default=None, help="Comma-separated assets, e.g. btc,eurusd,spx")
    parser.add_argument(
        "--intraday-map",
        type=str,
        default=None,
        help="Real-data CSV map: asset=path,asset2=path2",
    )
    parser.add_argument("--timezone", type=str, default="UTC", help="Timezone for day aggregation in real mode")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Quick mode (default)")
    mode.add_argument("--full", action="store_true", help="Full run mode")

    parser.add_argument("--no-lstm", action="store_true", help="Disable probabilistic LSTM model")
    parser.add_argument("--no-student-t", action="store_true", help="Disable Student-t HAR baseline")
    return parser


def _parse_intraday_map(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for item in raw.split(","):
        pair = item.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid intraday map item: {pair!r}. Expected asset=path.")
        asset, path = pair.split("=", 1)
        asset = asset.strip()
        path = path.strip()
        if not asset or not path:
            raise ValueError(f"Invalid intraday map item: {pair!r}. Expected asset=path.")
        out[asset] = path
    return out


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = PipelineConfig(seed=args.seed)
    if args.full:
        cfg.quick = False
    elif args.quick:
        cfg.quick = True

    cfg.data.data_source = args.data_source
    cfg.include_lstm = not args.no_lstm
    cfg.include_student_t_baseline = not args.no_student_t
    cfg.reporting.output_dir = args.output_dir
    cfg.reporting.run_name = args.run_name
    cfg.data.timezone = args.timezone

    assets_explicit = args.assets is not None
    if args.n_days is not None:
        cfg.data.n_days = int(args.n_days)
    if args.assets:
        cfg.data.assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cfg.data.intraday_file_map = _parse_intraday_map(args.intraday_map)

    if cfg.data.data_source == "real":
        if not cfg.data.intraday_file_map:
            raise ValueError("real mode requires --intraday-map with asset=path pairs")
        if not assets_explicit:
            cfg.data.assets = list(cfg.data.intraday_file_map.keys())
        missing_assets = [a for a in cfg.data.assets if a not in cfg.data.intraday_file_map]
        if missing_assets:
            raise ValueError(
                "Missing intraday files for assets in --assets: " + ",".join(missing_assets)
            )

    out = run_doctoral_pipeline(cfg)
    print(f"Run complete. Output root: {out['output_root']}")
    print(f"Summary: {out['summary_markdown']}")
    print(f"Manifest: {out['manifest_path']}")


if __name__ == "__main__":
    main()
