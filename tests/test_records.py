"""Record sampler: sizes, sources, disjointness, determinism, opaque ids (synthetic data)."""

import pandas as pd
import pytest

from significance.data import records


def _sets(train, dev, seed=5):
    return records.build_record_sets(
        train, dev, n_evaluation=6, n_selection=4, n_probe=2, seed=seed
    )


def test_record_sets_come_from_the_right_files_and_days(mind_frames):
    train, dev, _ = mind_frames
    sets = _sets(train, dev)
    assert set(sets) == {"evaluation", "selection", "probe"}
    assert len(sets["evaluation"]) == 6 and len(sets["selection"]) == 4 and len(sets["probe"]) == 2
    assert set(sets["evaluation"]["impression_id"]) <= set(dev["impression_id"])
    last_day = train["date"].max()
    for name in ("selection", "probe"):
        assert (sets[name]["date"] == last_day).all()
        assert set(sets[name]["split"]) == {"train"}
    assert set(sets["evaluation"]["split"]) == {"dev"}


def test_record_sets_are_disjoint_and_deterministic(mind_frames):
    train, dev, _ = mind_frames
    a, b = _sets(train, dev), _sets(train, dev)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
    ids = [set(a[n]["impression_id"]) for n in a]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    other = _sets(train, dev, seed=6)
    assert list(other["evaluation"]["impression_id"]) != list(a["evaluation"]["impression_id"])


def test_record_ids_are_positions_in_the_draw(mind_frames):
    train, dev, _ = mind_frames
    rec = _sets(train, dev)["evaluation"]
    assert list(rec["record_id"]) == list(range(len(rec)))
    assert list(rec.columns) == records.RECORD_COLUMNS
    assert "record_set" in rec and set(rec["record_set"]) == {"evaluation"}


def test_too_many_records_raise(mind_frames):
    _, dev, _ = mind_frames
    with pytest.raises(ValueError, match="pool"):
        records.sample_records(dev, len(dev) + 1, 1, "evaluation")


def test_save_and_load_round_trip(mind_frames, tmp_path):
    train, dev, _ = mind_frames
    sets = _sets(train, dev)
    records.save_record_sets(sets, tmp_path)
    back = records.load_records(tmp_path, "selection")
    pd.testing.assert_frame_equal(back, sets["selection"])
    with pytest.raises(ValueError, match="unknown"):
        records.load_records(tmp_path, "other")
