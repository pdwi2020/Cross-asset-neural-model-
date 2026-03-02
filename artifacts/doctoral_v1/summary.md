# Cross-Asset Doctoral Run Summary

## Configuration

- seed: `42`
- quick: `True`
- assets: `btc,eurusd,spx`
- n_days: `900`
- walkforward: `train=360,val=120,test=60,step=60`
- include_lstm: `True`
- var_alpha: `0.95`

## Leaderboard (Top Models)

- har_rv: RMSE=0.0015, MAE=0.0005, NLL=-4.5655, Cov95=0.969
- prob_har_gaussian: RMSE=0.0015, MAE=0.0005, NLL=-4.5655, Cov95=0.969
- var1_cross_asset: RMSE=0.0015, MAE=0.0005, NLL=-4.5069, Cov95=0.961
- naive_last_surface: RMSE=0.0020, MAE=0.0006, NLL=-4.2069, Cov95=0.963
- prob_lstm_gaussian: RMSE=0.0020, MAE=0.0006, NLL=-0.6904, Cov95=1.000

## DM Significance

- naive_last_surface vs har_rv: p=0.0427, Holm reject=False, BH reject=False
- prob_lstm_gaussian vs har_rv: p=0.0427, Holm reject=False, BH reject=False
- var1_cross_asset vs har_rv: p=0.4291, Holm reject=False, BH reject=False
- prob_har_gaussian vs har_rv: p=1.0000, Holm reject=False, BH reject=False

## Risk Calibration Snapshot

- prob_lstm_gaussian: exceed=0.000, Kupiec p=0.000, Christoffersen p=1.000
- naive_last_surface: exceed=0.024, Kupiec p=0.044, Christoffersen p=0.520
- har_rv: exceed=0.038, Kupiec p=0.317, Christoffersen p=0.527
- prob_har_gaussian: exceed=0.038, Kupiec p=0.317, Christoffersen p=0.527
- var1_cross_asset: exceed=0.039, Kupiec p=0.424, Christoffersen p=0.391

## Figures

- best_model_intervals: `figures/10_best_model_intervals.png`
- calibration_frontier: `figures/06_calibration_frontier.png`
- correlation_heatmap: `figures/03_correlation_heatmap.png`
- dm_significance: `figures/07_dm_significance.png`
- dynamic_correlations: `figures/04_dynamic_correlations.png`
- leaderboard_rmse: `figures/05_leaderboard_rmse.png`
- logrv_panels: `figures/02_logrv_panels.png`
- pit_histogram: `figures/11_pit_histogram.png`
- qq_residuals: `figures/12_qq_residuals.png`
- regime_timeline: `figures/01_regime_timeline.png`
- split_performance_drift: `figures/09_split_performance_drift.png`
- var_exceedance_heatmap: `figures/08_var_exceedance_heatmap.png`
