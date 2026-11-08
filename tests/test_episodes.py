"""Episode builder: previous decisions strictly earlier, real decisions apart (synthetic)."""

import pandas as pd
import pytest

from significance import episodes as E
from significance.data import records


def _build(mind_frames):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=8, n_selection=6, n_probe=2, seed=9)
    return train, dev, sets


def test_previous_decisions_are_history_then_earlier_days_only(mind_frames):
    train, dev, sets = _build(mind_frames)
    for name, beh in (("evaluation", dev), ("selection", train)):
        eps, _ = E.build_episodes(sets[name], train, dev)
        for ep, rec in zip(eps, sets[name].itertuples(), strict=True):
            row = beh.set_index("impression_id").loc[rec.impression_id]
            prev = list(ep.previous)
            if row["history"]:
                assert prev[0] == E.EarlierEpisode(None, tuple(row["history"]))
                prev = prev[1:]
            earlier = train[(train["user_id"] == rec.user_id) & (train["date"] < row["date"])]
            earlier = earlier.sort_values(["time", "impression_id"])
            assert [p.presented for p in prev] == [tuple(x) for x in earlier["presented"]]
            assert ep.earlier_presented == int(earlier["n_presented"].sum())
            assert ep.earlier_attended == int(earlier["n_clicked"].sum())
            assert ep.presented == tuple(row["presented"])


def test_selection_records_never_see_their_own_day(mind_frames):
    train, dev, sets = _build(mind_frames)
    eps, _ = E.build_episodes(sets["selection"], train, dev)
    last_day = train[train["date"] == train["date"].max()]
    last_day_sets = {tuple(p) for p in last_day["presented"]}
    for ep in eps:
        for p in ep.previous:
            if p.presented is not None:
                assert p.presented not in last_day_sets


def test_validation_impressions_are_never_previous_decisions(mind_frames):
    train, dev, sets = _build(mind_frames)
    eps, _ = E.build_episodes(sets["evaluation"], train, dev)
    dev_sets = {tuple(p) for p in dev["presented"]}
    train_sets = {tuple(p) for p in train["presented"]}
    for ep in eps:
        for p in ep.previous:
            if p.presented is not None:
                assert p.presented in train_sets and p.presented not in dev_sets


def test_real_decisions_are_a_separate_table(mind_frames):
    train, dev, sets = _build(mind_frames)
    eps, dec = E.build_episodes(sets["evaluation"], train, dev)
    assert list(dec.columns) == [
        "record_set",
        "record_id",
        "news_id",
        "attended",
        "seen_before",
        "attended_before",
    ]
    assert len(dec) == sum(ep.n for ep in eps)
    for ep in eps:
        assert not hasattr(ep, "decisions") and "attended" not in E.Episode.__dataclass_fields__
        rows = dec[dec["record_id"] == ep.record_id]
        rec = sets["evaluation"].set_index("record_id").loc[ep.record_id]
        imp = dev.set_index("impression_id").loc[rec["impression_id"]]
        assert list(rows["news_id"]) == list(imp["presented"])
        assert list(rows["attended"]) == list(imp["clicked"])


def test_seen_before_uses_previous_decisions_only(mind_frames):
    train, dev, sets = _build(mind_frames)
    eps, dec = E.build_episodes(sets["evaluation"], train, dev)
    for ep in eps:
        seen = {n for p in ep.previous for n in (p.presented or ()) + p.attended}
        att = {n for p in ep.previous for n in p.attended}
        rows = dec[dec["record_id"] == ep.record_id]
        assert list(rows["seen_before"]) == [n in seen for n in rows["news_id"]]
        assert list(rows["attended_before"]) == [n in att for n in rows["news_id"]]


def test_user_without_history_or_earlier_impressions(mind_frames):
    train, dev, _ = mind_frames
    row = dev[dev["user_id"] == "U6"].iloc[0]
    rec = pd.DataFrame(
        [
            {
                "record_set": "evaluation",
                "record_id": 0,
                "impression_id": row["impression_id"],
                "user_id": "U6",
                "time": row["time"],
                "date": row["date"],
                "split": "dev",
            }
        ]
    )
    eps, _ = E.build_episodes(rec, train, dev)
    assert all(p.presented is not None for p in eps[0].previous)  # no History session
    assert eps[0].history_clicks == 0


def test_mismatched_user_raises(mind_frames):
    train, dev, sets = _build(mind_frames)
    bad = sets["evaluation"].copy()
    bad.loc[0, "user_id"] = "U999"
    with pytest.raises(ValueError, match="user"):
        E.build_episodes(bad, train, dev)


def test_episode_storage_round_trip(mind_frames, tmp_path):
    train, dev, sets = _build(mind_frames)
    eps, _ = E.build_episodes(sets["selection"], train, dev)
    path = E.save_episodes(eps, tmp_path / "ep.jsonl")
    assert E.load_episodes(path) == eps
    assert any(ep.has_earlier_impressions for ep in eps)
