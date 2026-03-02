# Cross-Asset Doctoral Run Summary

## Configuration

- seed: `42`
- quick: `True`
- data_source: `synthetic`
- assets: `btc,eurusd,spx`
- n_days: `900`
- walkforward: `train=360,val=120,test=60,step=60`
- include_lstm: `True`
- include_har_j_baseline: `True`
- include_garch_baseline: `True`
- include_student_t_baseline: `True`
- var_alpha: `0.95`
- spa_benchmark: `naive_last_surface`
- economic_backtest: `True`
- transaction_cost_bps: `5.0`

## Leaderboard (Top Models)

- har_rv: RMSE=0.0015, MAE=0.0005, NLL=-4.5677, Cov95=0.969
- har_j_rv: RMSE=0.0015, MAE=0.0005, NLL=-4.5655, Cov95=0.969
- prob_har_gaussian: RMSE=0.0015, MAE=0.0005, NLL=-4.5655, Cov95=0.969
- prob_har_student_t: RMSE=0.0015, MAE=0.0005, NLL=-5.6136, Cov95=0.977
- garch11_qml: RMSE=0.0015, MAE=0.0005, NLL=-4.5683, Cov95=0.969

## DM Significance

- naive_last_surface vs har_rv: p=0.0431, Holm reject=False, BH reject=False
- prob_lstm_gaussian vs har_rv: p=0.0431, Holm reject=False, BH reject=False
- var1_cross_asset vs har_rv: p=0.4387, Holm reject=False, BH reject=False
- har_j_rv vs har_rv: p=1.0000, Holm reject=False, BH reject=False
- garch11_qml vs har_rv: p=1.0000, Holm reject=False, BH reject=False
- prob_har_gaussian vs har_rv: p=1.0000, Holm reject=False, BH reject=False
- prob_har_student_t vs har_rv: p=1.0000, Holm reject=False, BH reject=False

## SPA Test

- Global SPA p-value (benchmark `naive_last_surface`): `0.0165`
- var1_cross_asset: improvement=0.000002, one-sided p=0.0196
- har_j_rv: improvement=0.000002, one-sided p=0.0215
- prob_har_gaussian: improvement=0.000002, one-sided p=0.0215
- prob_har_student_t: improvement=0.000002, one-sided p=0.0215
- har_rv: improvement=0.000002, one-sided p=0.0218

## Model Confidence Set

- In-MCS models: har_j_rv, prob_har_gaussian, prob_har_student_t, har_rv, garch11_qml, var1_cross_asset

## Risk Calibration Snapshot

- prob_lstm_gaussian: exceed=0.000, Kupiec p=0.000, Christoffersen p=1.000
- naive_last_surface: exceed=0.024, Kupiec p=0.044, Christoffersen p=0.520
- prob_har_student_t: exceed=0.027, Kupiec p=0.080, Christoffersen p=0.467
- garch11_qml: exceed=0.036, Kupiec p=0.214, Christoffersen p=0.469
- har_j_rv: exceed=0.038, Kupiec p=0.317, Christoffersen p=0.527
- prob_har_gaussian: exceed=0.038, Kupiec p=0.317, Christoffersen p=0.527
- har_rv: exceed=0.039, Kupiec p=0.378, Christoffersen p=0.557
- var1_cross_asset: exceed=0.039, Kupiec p=0.424, Christoffersen p=0.391

## Economic Backtest Snapshot

- var1_cross_asset: Sharpe=0.574, AnnRet=0.062, AnnVol=0.108, MDD=-0.085
- naive_last_surface: Sharpe=0.317, AnnRet=0.072, AnnVol=0.229, MDD=-0.190
- prob_lstm_gaussian: Sharpe=0.317, AnnRet=0.072, AnnVol=0.229, MDD=-0.190
- har_rv: Sharpe=0.096, AnnRet=0.010, AnnVol=0.108, MDD=-0.083
- har_j_rv: Sharpe=0.072, AnnRet=0.008, AnnVol=0.108, MDD=-0.092

## Figures

- best_model_intervals: `figures/10_best_model_intervals.png`
- calibration_frontier: `figures/06_calibration_frontier.png`
- correlation_heatmap: `figures/03_correlation_heatmap.png`
- dm_significance: `figures/07_dm_significance.png`
- dynamic_correlations: `figures/04_dynamic_correlations.png`
- economic_equity_curve: `figures/16_economic_equity_curve.png`
- economic_sharpe: `figures/15_economic_sharpe.png`
- leaderboard_rmse: `figures/05_leaderboard_rmse.png`
- logrv_panels: `figures/02_logrv_panels.png`
- model_confidence_set: `figures/14_model_confidence_set.png`
- pit_histogram: `figures/11_pit_histogram.png`
- qq_residuals: `figures/12_qq_residuals.png`
- regime_timeline: `figures/01_regime_timeline.png`
- spa_improvements: `figures/13_spa_improvements.png`
- split_performance_drift: `figures/09_split_performance_drift.png`
- var_exceedance_heatmap: `figures/08_var_exceedance_heatmap.png`
