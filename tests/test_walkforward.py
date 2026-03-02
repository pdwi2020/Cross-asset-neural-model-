from cross_asset_research.walkforward import generate_walkforward_splits


def test_walkforward_no_overlap_and_ordering() -> None:
    splits = generate_walkforward_splits(
        n_samples=420,
        train_size=200,
        val_size=80,
        test_size=40,
        step_size=40,
    )

    assert len(splits) > 0
    for s in splits:
        assert s.train_idx.max() < s.val_idx.min()
        assert s.val_idx.max() < s.test_idx.min()
        assert len(set(s.train_idx) & set(s.val_idx)) == 0
        assert len(set(s.val_idx) & set(s.test_idx)) == 0
