"""Configuration objects for cross-asset research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DataConfig:
    """Data source settings for synthetic or real intraday ingestion."""

    assets: List[str] = field(default_factory=lambda: ["btc", "eurusd", "spx"])
    data_source: str = "synthetic"  # synthetic | real
    n_days: int = 1800
    start_date: str = "2016-01-01"
    timezone: str = "UTC"
    jump_z: float = 4.0
    min_obs_per_day: int = 30
    intraday_file_map: Dict[str, str] = field(default_factory=dict)
    regime_transition: List[List[float]] = field(
        default_factory=lambda: [
            [0.95, 0.04, 0.01],
            [0.07, 0.88, 0.05],
            [0.04, 0.10, 0.86],
        ]
    )


@dataclass
class FeatureConfig:
    """Feature engineering settings."""

    corr_window: int = 30
    beta_window: int = 30
    weekly_lag: int = 5
    monthly_lag: int = 22


@dataclass
class WalkForwardConfig:
    """Walk-forward split settings."""

    train_size: int = 700
    val_size: int = 180
    test_size: int = 90
    step_size: int = 90


@dataclass
class ModelConfig:
    """Model training settings."""

    sequence_length: int = 30
    lstm_hidden: int = 32
    lstm_layers: int = 1
    lstm_dropout: float = 0.1
    lstm_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3


@dataclass
class RiskConfig:
    """Risk backtesting settings."""

    var_alpha: float = 0.95
    var_horizon_days: int = 1


@dataclass
class ReportingConfig:
    """Reporting and artifact settings."""

    output_dir: str = "artifacts"
    run_name: str = "latest"
    dpi: int = 180
    save_predictions_csv: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    seed: int = 42
    quick: bool = True
    alpha: float = 0.05
    include_lstm: bool = True
    include_student_t_baseline: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def apply_quick_mode(self) -> None:
        """Reduce runtime for smoke/CI workflows."""

        if not self.quick:
            return
        if self.data.data_source == "synthetic":
            self.data.n_days = 900
        self.walkforward.train_size = 360
        self.walkforward.val_size = 120
        self.walkforward.test_size = 60
        self.walkforward.step_size = 60
        self.model.lstm_epochs = 4
        self.model.lstm_hidden = 24
