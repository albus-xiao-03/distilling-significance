"""Expanded-dataset views: id map, re-keying and view assembly (synthetic data, fake client)."""

import json

import pandas as pd
import pytest

from fakes import FakeClient
from significance import episodes as E
from significance import expansion as X
from significance import runner
from significance.data import records
from significance.teacher import api


def _frames(mind_frames):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=4, n_selection=3, n_probe=1, seed=2)
    sets.update(records.build_extra_record_sets(train, dev, sets, n_total=24, seed=5))
    return train, dev, sets


def test_id_map_keeps_evaluation_ids_and_numbers_the_rest_in_order(mind_frames):
    _, _, sets = _frames(mind_frames)
    ids = X.id_map(sets)
    ev = ids[ids["record_set"] == "evaluation"]
    assert (ev["dataset_id"] == ev["record_id"]).all()
    assert sorted(ids["dataset_id"]) == list(range(sum(len(s) for s in sets.values())))
    order = ids.groupby("record_set")["dataset_id"].min().sort_values().index.tolist()
    assert order == [name for name, _ in X.SOURCES]
    assert set(ids.loc[ids["record_set"] == "selection", "split"]) == {"train"}
    with pytest.raises(ValueError, match="record ids"):
        X.id_map({**sets, "probe": sets["probe"].assign(record_id=7)})


def test_rekey_frames_and_episodes(mind_frames):
    train, dev, sets = _frames(mind_frames)
    ids = X.id_map(sets)
    eps, dec = E.build_episodes(sets["selection"], train, dev)
    out = X.rekey(dec, ids, "dataset")
    assert set(out["record_set"]) == {"dataset"} and list(out.columns) == list(dec.columns)
    sel = ids[ids["record_set"] == "selection"].set_index("record_id")["dataset_id"]
    assert out["record_id"].tolist() == dec["record_id"].map(sel).tolist()
    re = X.rekey_episodes(eps, ids, "dataset")
    assert [e.record_id for e in re] == [int(sel[e.record_id]) for e in eps]
    assert all(
        a.previous == b.previous and a.presented == b.presented
        for a, b in zip(eps, re, strict=True)
    )
    with pytest.raises(ValueError, match="missing"):
        X.rekey(dec.assign(record_set="other"), ids, "dataset")


def test_build_views_assembles_labels_without_new_calls(mind_frames, tmp_path):
    train, dev, sets = _frames(mind_frames)
    records.save_record_sets(sets, tmp_path)
    client = FakeClient()
    with api.ResponseCache(tmp_path / "c.sqlite") as cache:
        teacher = api.Teacher(client, "cat", cache=cache, sleep=lambda s: None)
        for name, rec in sets.items():
            eps, dec = E.build_episodes(rec, train, dev)
            E.save_episodes(eps, E.episodes_dir(tmp_path) / f"{name}.jsonl")
            dec.to_parquet(E.episodes_dir(tmp_path) / f"decisions_{name}.parquet", index=False)
            rows, facts = runner.run_variant(teacher, eps, "V1o", workers=2)
            runner.save_predictions(
                rows, facts, runner.predictions_path(tmp_path, name, "V1o", False)
            )
            if name in ("evaluation", "evaluation_extra"):
                rows, facts = runner.run_variant(teacher, eps, "V1o", empty_history=True, workers=2)
                runner.save_predictions(
                    rows, facts, runner.predictions_path(tmp_path, name, "V1o", True)
                )
    calls = len(client.calls)
    facts = X.build_views(tmp_path, "V1o")
    assert len(client.calls) == calls  # assembling makes no call
    total = sum(len(s) for s in sets.values())
    assert facts["dataset"]["records"] == total
    assert facts["dataset_eval"]["records"] == total - len(sets["selection"])
    assert facts["dataset_validation"]["records"] == len(sets["evaluation"]) + len(
        sets["evaluation_extra"]
    )
    full = pd.read_parquet(runner.predictions_path(tmp_path, "dataset", "V1o", False))
    ids = pd.read_parquet(tmp_path / "dataset" / "id_map.parquet")
    for name in sets:  # the labels are the source runs' labels, row for row
        src = pd.read_parquet(runner.predictions_path(tmp_path, name, "V1o", False))
        sub = ids[ids["record_set"] == name].set_index("record_id")["dataset_id"]
        mine = full[full["record_id"].isin(sub.values)].sort_values(["record_id", "news_id"])
        theirs = src.assign(record_id=src["record_id"].map(sub)).sort_values(
            ["record_id", "news_id"]
        )
        assert mine["attention_rank"].tolist() == theirs["attention_rank"].tolist()
        assert mine["predicted_attention"].tolist() == theirs["predicted_attention"].tolist()
    eps, dec, rec = runner.load_set(tmp_path, "dataset_eval")
    assert len(eps) == len(rec) == facts["dataset_eval"]["records"]
    assert not set(rec["record_id"]) & set(ids.loc[ids["record_set"] == "selection", "dataset_id"])
    assert runner.predictions_path(tmp_path, "dataset_validation", "V1o", True).is_file()
    assert json.loads((tmp_path / "dataset" / "views.json").read_text()).keys() == X.VIEWS.keys()
