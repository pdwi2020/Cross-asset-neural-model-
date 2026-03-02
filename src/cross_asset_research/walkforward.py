"""Walk-forward split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class WalkForwardSplit:
    """One walk-forward split with train/validation/test indices."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def generate_walkforward_splits(
    n_samples: int,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
) -> List[WalkForwardSplit]:
    """Create leakage-free walk-forward windows."""

    splits: List[WalkForwardSplit] = []
    start = 0
    while True:
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > n_samples:
            break
        splits.append(
            WalkForwardSplit(
                train_idx=np.arange(start, train_end),
                val_idx=np.arange(train_end, val_end),
                test_idx=np.arange(val_end, test_end),
            )
        )
        start += step_size
    return splits


def summarize_splits(splits: Iterable[WalkForwardSplit]) -> List[dict]:
    """Serialize split boundaries for reporting."""

    out: List[dict] = []
    for i, split in enumerate(splits):
        out.append(
            {
                "split_id": i,
                "train_start": int(split.train_idx.min()),
                "train_end": int(split.train_idx.max()),
                "val_start": int(split.val_idx.min()),
                "val_end": int(split.val_idx.max()),
                "test_start": int(split.test_idx.min()),
                "test_end": int(split.test_idx.max()),
            }
        )
    return out
