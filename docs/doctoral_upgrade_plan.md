# Doctoral Upgrade Plan (Implemented + Next)

## Implemented in this repo

1. Reproducible package architecture (`src/` + CLI + tests)
2. Regime-switching cross-asset simulation with contagion and jumps
3. Feature stack for HAR/cross-asset structure (lags, rolling correlation, beta, spillover)
4. Leakage-free walk-forward framework
5. Multi-model benchmark suite (Naive, HAR, VAR, ProbHAR, ProbLSTM)
6. Probabilistic evaluation (NLL + interval diagnostics)
7. Formal significance testing (Diebold-Mariano + Holm/BH)
8. VaR/ES backtesting (Kupiec UC + Christoffersen independence)
9. Advanced report generation (12 figures, multiple tables, manifest)
10. End-to-end test coverage including smoke E2E run

## High-ROI next steps

1. Replace synthetic generator with real intraday ingestion pipeline
2. Add heavy-tail likelihoods (Student-t / skew-t / Gaussian mixture)
3. Add regime-conditioned experts or switching state-space models
4. Add feature attribution and stability analysis (SHAP + temporal drift)
5. Translate risk forecasts into portfolio controls and trading constraints
6. Add experiment tracking (MLflow/W&B) for audit-grade provenance
7. Add CI/CD for automated backtests and artifact publishing

## Publication path

- Target a methods + empirical paper structure:
  - theory and probabilistic formulation
  - data engineering and market microstructure controls
  - model architecture and training protocol
  - walk-forward results and significance tests
  - risk backtesting and calibration analysis
  - robustness, ablations, and failure modes
