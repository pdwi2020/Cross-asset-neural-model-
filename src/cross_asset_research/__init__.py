"""Cross-asset doctoral research package."""

from .config import PipelineConfig


def run_doctoral_pipeline(*args, **kwargs):
    """Lazy import to avoid side effects when invoking CLI module entrypoints."""

    from .pipeline import run_doctoral_pipeline as _run_doctoral_pipeline

    return _run_doctoral_pipeline(*args, **kwargs)

__all__ = ["PipelineConfig", "run_doctoral_pipeline"]
