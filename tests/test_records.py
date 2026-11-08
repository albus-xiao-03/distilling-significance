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


def _expanded(mind_frames, n_total=24, seed=5):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=4, n_selection=3, n_probe=1, seed=2)
    extra = records.build_extra_record_sets(train, dev, sets, n_total=n_total, seed=seed)
    return train, dev, sets, extra


def test_split_counts_largest_remainder():
    assert records.split_counts(25_000, {"train": 156_965, "dev": 73_152}) == {
        "train": 17_053,
        "dev": 7_947,
    }
    assert sum(records.split_counts(10, {"a": 1, "b": 1, "c": 1}).values()) == 10
    with pytest.raises(ValueError):
        records.split_counts(5, {"a": 0})


def test_extra_records_complete_a_proportional_dataset(mind_frames):
    train, dev, sets, extra = _expanded(mind_frames)
    files = records.split_counts(24, {"train": len(train), "dev": len(dev)})
    n_dev = len(sets["evaluation"]) + len(extra["evaluation_extra"])
    n_train = len(sets["selection"]) + len(sets["probe"]) + len(extra["training_extra"])
    assert (n_train, n_dev) == (files["train"], files["dev"]) and n_train + n_dev == 24
    everything = pd.concat([*sets.values(), *extra.values()])
    assert not everything["impression_id"].duplicated().any()  # the two files' ids differ
    assert set(extra["evaluation_extra"]["split"]) == {"dev"}
    assert set(extra["training_extra"]["split"]) == {"train"}
    quotas = records.split_counts(files["train"], train.groupby("date").size().to_dict())
    train_part = everything[everything["split"] == "train"]
    assert train_part.groupby("date").size().to_dict() == quotas


def test_extra_records_are_deterministic_and_reuse_existing(mind_frames):
    _, _, sets, a = _expanded(mind_frames)
    _, _, _, b = _expanded(mind_frames)
    for name in records.EXTRA_SETS:
        pd.testing.assert_frame_equal(a[name], b[name])
        assert list(a[name]["record_id"]) == list(range(len(a[name])))
        assert set(a[name]["record_set"]) == {name}
    old = set(pd.concat(sets.values())["impression_id"])
    for name in records.EXTRA_SETS:
        assert not old & set(a[name]["impression_id"])


def test_extra_records_refuse_impossible_quotas(mind_frames):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=4, n_selection=3, n_probe=1, seed=2)
    with pytest.raises(ValueError, match="exceed the quota"):
        records.build_extra_record_sets(train, dev, sets, n_total=6)
    few_dev = {**sets, "evaluation": sets["evaluation"].iloc[:1]}
    with pytest.raises(ValueError, match="exceed the quota"):
        records.build_extra_record_sets(train, dev, few_dev, n_total=8)
