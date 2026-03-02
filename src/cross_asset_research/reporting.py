"""Reporting helpers: tables, figures, and markdown summary artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, probplot, t


def prepare_report_dirs(output_dir: str, run_name: str) -> Dict[str, Path]:
    """Create output directory structure for one run."""

    root = Path(output_dir).expanduser().resolve() / run_name
    figs = root / "figures"
    tables = root / "tables"
    root.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return {"root": root, "figures": figs, "tables": tables}


def save_tables(
    *,
    leaderboard_df: pd.DataFrame,
    asset_metrics_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    spa_df: pd.DataFrame,
    mcs_df: pd.DataFrame,
    mcs_elim_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    economic_df: pd.DataFrame,
    economic_pnl_df: pd.DataFrame,
    split_perf_df: pd.DataFrame,
    split_boundaries_df: pd.DataFrame,
    out_tables_dir: Path,
) -> Dict[str, Path]:
    """Persist all tabular artifacts as CSV."""

    outputs: Dict[str, Path] = {}

    outputs["leaderboard"] = out_tables_dir / "leaderboard.csv"
    leaderboard_df.to_csv(outputs["leaderboard"], index=False)

    outputs["asset_metrics"] = out_tables_dir / "asset_metrics.csv"
    asset_metrics_df.to_csv(outputs["asset_metrics"], index=False)

    outputs["dm_tests"] = out_tables_dir / "dm_tests.csv"
    dm_df.to_csv(outputs["dm_tests"], index=False)

    outputs["spa_tests"] = out_tables_dir / "spa_tests.csv"
    spa_df.to_csv(outputs["spa_tests"], index=False)

    outputs["model_confidence_set"] = out_tables_dir / "model_confidence_set.csv"
    mcs_df.to_csv(outputs["model_confidence_set"], index=False)

    outputs["model_confidence_set_elimination"] = out_tables_dir / "model_confidence_set_elimination.csv"
    mcs_elim_df.to_csv(outputs["model_confidence_set_elimination"], index=False)

    outputs["risk_backtests"] = out_tables_dir / "risk_backtests.csv"
    risk_df.to_csv(outputs["risk_backtests"], index=False)

    outputs["economic_backtests"] = out_tables_dir / "economic_backtests.csv"
    economic_df.to_csv(outputs["economic_backtests"], index=False)

    outputs["economic_portfolio_pnl"] = out_tables_dir / "economic_portfolio_pnl.csv"
    economic_pnl_df.to_csv(outputs["economic_portfolio_pnl"], index=True)

    outputs["split_performance"] = out_tables_dir / "split_performance.csv"
    split_perf_df.to_csv(outputs["split_performance"], index=False)

    outputs["split_boundaries"] = out_tables_dir / "split_boundaries.csv"
    split_boundaries_df.to_csv(outputs["split_boundaries"], index=False)

    return outputs


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_advanced_figures(
    *,
    raw_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    assets: Iterable[str],
    leaderboard_df: pd.DataFrame,
    asset_metrics_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    spa_df: pd.DataFrame,
    mcs_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    economic_df: pd.DataFrame,
    economic_pnl_df: pd.DataFrame,
    split_perf_df: pd.DataFrame,
    predictions_by_model: Dict[str, Dict[str, pd.DataFrame]],
    best_model: str,
    out_fig_dir: Path,
    dpi: int = 180,
) -> Dict[str, Path]:
    """Render publication-style figures for research diagnostics."""

    sns.set_theme(style="whitegrid", context="talk")
    assets = list(assets)
    outputs: Dict[str, Path] = {}

    if "regime_id" in raw_df.columns:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(raw_df.index, raw_df["regime_id"], color="#1f77b4", lw=1.2)
        ax.set_title("Latent Regime State Timeline")
        ax.set_ylabel("Regime ID")
        ax.set_xlabel("Date")
        path = out_fig_dir / "01_regime_timeline.png"
        _save_figure(fig, path, dpi)
        outputs["regime_timeline"] = path

    fig, axs = plt.subplots(len(assets), 1, figsize=(14, 3.5 * len(assets)), sharex=True)
    if len(assets) == 1:
        axs = [axs]
    for ax, asset in zip(axs, assets):
        ax.plot(raw_df.index, raw_df[f"log_{asset}_rv"], color="#d62728", lw=1.0)
        ax.set_title(f"{asset.upper()} Log Realized Volatility")
        ax.set_ylabel("log(1 + RV)")
    axs[-1].set_xlabel("Date")
    path = out_fig_dir / "02_logrv_panels.png"
    _save_figure(fig, path, dpi)
    outputs["logrv_panels"] = path

    ret_cols = [f"{a}_ret" for a in assets if f"{a}_ret" in raw_df.columns]
    if ret_cols:
        if "regime_id" in raw_df.columns:
            stress_df = raw_df.loc[raw_df["regime_id"] == raw_df["regime_id"].max(), ret_cols]
            use_df = stress_df if len(stress_df) >= 20 else raw_df[ret_cols]
        else:
            use_df = raw_df[ret_cols]

        corr = use_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Cross-Asset Return Correlation (Stress Regime Focus)")
        path = out_fig_dir / "03_correlation_heatmap.png"
        _save_figure(fig, path, dpi)
        outputs["correlation_heatmap"] = path

    corr_cols = [c for c in engineered_df.columns if c.startswith("corr_")][:8]
    if corr_cols:
        fig, ax = plt.subplots(figsize=(14, 6))
        for c in corr_cols:
            ax.plot(engineered_df.index, engineered_df[c], lw=1.1, label=c)
        ax.set_title("Dynamic Rolling Correlations")
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.legend(ncol=2, fontsize=8)
        path = out_fig_dir / "04_dynamic_correlations.png"
        _save_figure(fig, path, dpi)
        outputs["dynamic_correlations"] = path

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=leaderboard_df,
        x="model",
        y="aggregate_rmse",
        hue="model",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Model Leaderboard by Aggregate RMSE")
    ax.set_xlabel("Model")
    ax.set_ylabel("Aggregate RMSE")
    ax.tick_params(axis="x", rotation=25)
    path = out_fig_dir / "05_leaderboard_rmse.png"
    _save_figure(fig, path, dpi)
    outputs["leaderboard_rmse"] = path

    if not asset_metrics_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=asset_metrics_df,
            x="avg_interval_width_95",
            y="coverage_95",
            hue="model",
            style="asset",
            s=130,
            ax=ax,
        )
        ax.axhline(0.95, ls="--", color="black", lw=1.0)
        ax.set_title("Calibration Frontier: Coverage vs Interval Width")
        ax.set_xlabel("Average 95% Interval Width")
        ax.set_ylabel("Observed Coverage")
        path = out_fig_dir / "06_calibration_frontier.png"
        _save_figure(fig, path, dpi)
        outputs["calibration_frontier"] = path

    if not dm_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        score = -np.log10(np.maximum(dm_df["p_value"].to_numpy(dtype=float), 1e-10))
        sns.barplot(
            x=dm_df["model"],
            y=score,
            hue=dm_df["model"],
            palette="rocket",
            legend=False,
            ax=ax,
        )
        ax.axhline(-np.log10(0.05), ls="--", color="black", lw=1.0)
        ax.set_title("Diebold-Mariano Significance vs Baseline")
        ax.set_xlabel("Model")
        ax.set_ylabel("-log10(p-value)")
        ax.tick_params(axis="x", rotation=20)
        path = out_fig_dir / "07_dm_significance.png"
        _save_figure(fig, path, dpi)
        outputs["dm_significance"] = path

    if not spa_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=spa_df,
            x="model",
            y="mean_loss_improvement_vs_benchmark",
            hue="model",
            palette="crest",
            legend=False,
            ax=ax,
        )
        ax.axhline(0.0, ls="--", color="black", lw=1.0)
        spa_p = float(spa_df["spa_global_p"].iloc[0])
        ax.set_title(f"SPA Improvements vs Benchmark (global p={spa_p:.3f})")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean loss improvement")
        ax.tick_params(axis="x", rotation=20)
        path = out_fig_dir / "13_spa_improvements.png"
        _save_figure(fig, path, dpi)
        outputs["spa_improvements"] = path

    if not mcs_df.empty:
        plot_df = mcs_df.copy()
        plot_df["in_mcs_flag"] = plot_df["in_mcs"].astype(int)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=plot_df,
            x="model",
            y="mean_loss",
            hue="in_mcs",
            style="in_mcs",
            s=150,
            ax=ax,
        )
        ax.set_title("Model Confidence Set Membership")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean loss")
        ax.tick_params(axis="x", rotation=20)
        path = out_fig_dir / "14_model_confidence_set.png"
        _save_figure(fig, path, dpi)
        outputs["model_confidence_set"] = path

    if not risk_df.empty:
        pivot = risk_df.pivot(index="asset", columns="model", values="observed_exceed_rate")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        ax.set_title("Observed VaR Exceedance Rates")
        path = out_fig_dir / "08_var_exceedance_heatmap.png"
        _save_figure(fig, path, dpi)
        outputs["var_exceedance_heatmap"] = path

    if not economic_df.empty:
        port = economic_df[economic_df["asset"] == "portfolio_equal_weight"].copy()
        if not port.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=port.sort_values("sharpe_net", ascending=False),
                x="model",
                y="sharpe_net",
                hue="model",
                palette="magma",
                legend=False,
                ax=ax,
            )
            ax.axhline(0.0, ls="--", color="black", lw=1.0)
            ax.set_title("Economic Backtest: Net Sharpe by Model")
            ax.set_xlabel("Model")
            ax.set_ylabel("Net Sharpe")
            ax.tick_params(axis="x", rotation=20)
            path = out_fig_dir / "15_economic_sharpe.png"
            _save_figure(fig, path, dpi)
            outputs["economic_sharpe"] = path

    if not economic_pnl_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        eq = (1.0 + economic_pnl_df.fillna(0.0)).cumprod()
        for c in eq.columns:
            ax.plot(eq.index, eq[c], lw=1.2, label=c)
        ax.set_title("Economic Backtest: Cumulative Net Portfolio Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity curve")
        ax.legend(ncol=2, fontsize=8)
        path = out_fig_dir / "16_economic_equity_curve.png"
        _save_figure(fig, path, dpi)
        outputs["economic_equity_curve"] = path

    if not split_perf_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=split_perf_df, x="split_id", y="aggregate_rmse", hue="model", marker="o", ax=ax)
        ax.set_title("Walk-Forward Performance Drift")
        ax.set_xlabel("Split ID")
        ax.set_ylabel("Aggregate RMSE")
        path = out_fig_dir / "09_split_performance_drift.png"
        _save_figure(fig, path, dpi)
        outputs["split_performance_drift"] = path

    if best_model in predictions_by_model:
        pred = predictions_by_model[best_model]
        y = pred["y_true"].sort_index()
        mu = pred["mu"].sort_index()
        sigma = pred["sigma"].sort_index()
        distribution = str(pred.get("distribution", "gaussian"))
        dof_df = pred.get("dof")

        asset = assets[0]
        if distribution == "student_t" and dof_df is not None and asset in dof_df.columns:
            nu = np.maximum(dof_df[asset].to_numpy(dtype=float), 2.1)
            q = t.ppf(0.975, df=nu)
            lower = mu[asset].to_numpy(dtype=float) - q * sigma[asset].to_numpy(dtype=float)
            upper = mu[asset].to_numpy(dtype=float) + q * sigma[asset].to_numpy(dtype=float)
            lower = pd.Series(lower, index=mu.index)
            upper = pd.Series(upper, index=mu.index)
            standardized = (y[asset] - mu[asset]) / np.maximum(sigma[asset], 1e-6)
            pit = t.cdf(standardized.to_numpy(dtype=float), df=nu)
        else:
            z = norm.ppf(0.975)
            lower = mu[asset] - z * sigma[asset]
            upper = mu[asset] + z * sigma[asset]
            standardized = (y[asset] - mu[asset]) / np.maximum(sigma[asset], 1e-6)
            pit = norm.cdf(standardized.to_numpy(dtype=float))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(y.index, y[asset], label="realized", color="#1f77b4", lw=1.2)
        ax.plot(mu.index, mu[asset], label="predicted mean", color="#d62728", lw=1.2)
        ax.fill_between(mu.index, lower, upper, color="#d62728", alpha=0.18, label="95% interval")
        ax.set_title(
            f"Best Model Forecast Intervals ({best_model}, {asset.upper()}, {distribution})"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("log(1 + RV)")
        ax.legend(loc="best")
        path = out_fig_dir / "10_best_model_intervals.png"
        _save_figure(fig, path, dpi)
        outputs["best_model_intervals"] = path

        zscore = standardized

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(pit, bins=20, density=True, color="#17becf", alpha=0.8, edgecolor="black")
        ax.axhline(1.0, ls="--", color="black", lw=1.0)
        ax.set_title(f"PIT Histogram ({best_model}, {asset.upper()})")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        path = out_fig_dir / "11_pit_histogram.png"
        _save_figure(fig, path, dpi)
        outputs["pit_histogram"] = path

        fig, ax = plt.subplots(figsize=(10, 5))
        probplot(zscore.dropna(), dist="norm", plot=ax)
        ax.set_title(f"Standardized Residual Q-Q Plot ({best_model}, {asset.upper()})")
        path = out_fig_dir / "12_qq_residuals.png"
        _save_figure(fig, path, dpi)
        outputs["qq_residuals"] = path

    return outputs


def write_summary_markdown(
    *,
    output_root: Path,
    config_summary: Dict[str, object],
    leaderboard_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    spa_df: pd.DataFrame,
    mcs_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    economic_df: pd.DataFrame,
    figure_paths: Dict[str, Path],
) -> Path:
    """Write a concise run summary markdown report."""

    lines: List[str] = []
    lines.append("# Cross-Asset Doctoral Run Summary")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for k, v in config_summary.items():
        lines.append(f"- {k}: `{v}`")

    lines.append("")
    lines.append("## Leaderboard (Top Models)")
    lines.append("")
    top = leaderboard_df.head(5).copy()
    for _, row in top.iterrows():
        lines.append(
            "- "
            + f"{row['model']}: RMSE={row['aggregate_rmse']:.4f}, MAE={row['aggregate_mae']:.4f}, "
            + f"NLL={row['aggregate_nll']:.4f}, Cov95={row['aggregate_coverage_95']:.3f}"
        )

    if not dm_df.empty:
        lines.append("")
        lines.append("## DM Significance")
        lines.append("")
        for _, row in dm_df.iterrows():
            lines.append(
                "- "
                + f"{row['model']} vs {row['baseline']}: p={row['p_value']:.4f}, "
                + f"Holm reject={bool(row.get('holm_reject_h0', False))}, "
                + f"BH reject={bool(row.get('bh_reject_h0', False))}"
            )

    if not spa_df.empty:
        lines.append("")
        lines.append("## SPA Test")
        lines.append("")
        spa_p = float(spa_df["spa_global_p"].iloc[0])
        benchmark = str(spa_df["benchmark"].iloc[0])
        lines.append(f"- Global SPA p-value (benchmark `{benchmark}`): `{spa_p:.4f}`")
        top_spa = spa_df.sort_values("model_one_sided_p").head(5)
        for _, row in top_spa.iterrows():
            lines.append(
                "- "
                + f"{row['model']}: improvement={row['mean_loss_improvement_vs_benchmark']:.6f}, "
                + f"one-sided p={row['model_one_sided_p']:.4f}"
            )

    if not mcs_df.empty:
        lines.append("")
        lines.append("## Model Confidence Set")
        lines.append("")
        in_set = mcs_df[mcs_df["in_mcs"] == True]  # noqa: E712
        lines.append("- In-MCS models: " + ", ".join(in_set["model"].astype(str).tolist()))

    if not risk_df.empty:
        lines.append("")
        lines.append("## Risk Calibration Snapshot")
        lines.append("")
        agg = (
            risk_df.groupby("model", as_index=False)
            .agg(
                observed_exceed_rate=("observed_exceed_rate", "mean"),
                kupiec_p_value=("kupiec_p_value", "mean"),
                christoffersen_p_value=("christoffersen_p_value", "mean"),
            )
            .sort_values("observed_exceed_rate")
        )
        for _, row in agg.iterrows():
            lines.append(
                "- "
                + f"{row['model']}: exceed={row['observed_exceed_rate']:.3f}, "
                + f"Kupiec p={row['kupiec_p_value']:.3f}, Christoffersen p={row['christoffersen_p_value']:.3f}"
            )

    if not economic_df.empty:
        lines.append("")
        lines.append("## Economic Backtest Snapshot")
        lines.append("")
        port = economic_df[economic_df["asset"] == "portfolio_equal_weight"].copy()
        if not port.empty:
            for _, row in port.sort_values("sharpe_net", ascending=False).head(5).iterrows():
                lines.append(
                    "- "
                    + f"{row['model']}: Sharpe={row['sharpe_net']:.3f}, "
                    + f"AnnRet={row['ann_return_net']:.3f}, "
                    + f"AnnVol={row['ann_vol_net']:.3f}, MDD={row['max_drawdown_net']:.3f}"
                )

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name, path in sorted(figure_paths.items()):
        rel = path.relative_to(output_root)
        lines.append(f"- {name}: `{rel}`")

    out = output_root / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
