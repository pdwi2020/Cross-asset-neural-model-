# Doctoral Upgrade Plan (Implemented + Next)

## Implemented in this repo

1. Reproducible package architecture (`src/` + CLI + tests)
2. Regime-switching cross-asset simulation with contagion and jumps
3. Feature stack for HAR/cross-asset structure (lags, rolling correlation, beta, spillover)
4. Leakage-free walk-forward framework
5. Multi-model benchmark suite (Naive, HAR, VAR, ProbHAR Gaussian, ProbHAR Student-t, ProbLSTM)
6. Probabilistic evaluation (NLL + interval diagnostics)
7. Formal significance testing (Diebold-Mariano + Holm/BH)
8. VaR/ES backtesting (Kupiec UC + Christoffersen independence)
9. Advanced report generation (12 figures, multiple tables, manifest)
10. Real-data ingestion (intraday minute-bars -> daily realized measures)
11. Distribution-aware evaluation and risk backtesting (Gaussian vs Student-t)
12. End-to-end test coverage including synthetic and real-data smoke runs

## High-ROI next steps

1. Add richer heavy-tail families (skew-t / Gaussian-mixture / EVT tail splice)
2. Add regime-conditioned experts or switching state-space models
3. Add feature attribution and stability analysis (SHAP + temporal drift)
4. Translate risk forecasts into portfolio controls and trading constraints
5. Add experiment tracking (MLflow/W&B) for audit-grade provenance
6. Add CI/CD for automated backtests and artifact publishing
7. Add real market data adapters (Polygon/Alpaca/Crypto exchanges) with cache + schema contracts

## Publication path

- Target a methods + empirical paper structure:
  - theory and probabilistic formulation
  - data engineering and market microstructure controls
  - model architecture and training protocol
  - walk-forward results and significance tests
  - risk backtesting and calibration analysis
  - robustness, ablations, and failure modes
