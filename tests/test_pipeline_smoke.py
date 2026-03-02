from pathlib import Path

from cross_asset_research.config import PipelineConfig
from cross_asset_research.pipeline import run_doctoral_pipeline


def test_pipeline_smoke_generates_artifacts(tmp_path: Path) -> None:
    cfg = PipelineConfig(seed=123, quick=True, include_lstm=False)
    cfg.reporting.output_dir = str(tmp_path)
    cfg.reporting.run_name = "smoke"

    out = run_doctoral_pipeline(cfg)

    assert out["leaderboard"].shape[0] >= 3
    assert out["manifest_path"].exists()
    assert out["summary_markdown"].exists()

    fig_dir = tmp_path / "smoke" / "figures"
    table_dir = tmp_path / "smoke" / "tables"
    assert fig_dir.exists()
    assert table_dir.exists()
    assert len(list(fig_dir.glob("*.png"))) >= 8
