"""Runner: approval guard, request estimate, parallel run, variant choice (fake client)."""

import pandas as pd
import pytest

from fakes import FakeClient
from significance import episodes as E
from significance import runner
from significance.data import records
from significance.teacher import api
from significance.teacher.decode import requests_needed


def _setup(mind_frames):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=8, n_selection=6, n_probe=2, seed=2)
    eps, dec = E.build_episodes(sets["selection"], train, dev)
    return eps, dec, sets["selection"]


def test_full_runs_need_approval():
    for name in ("selection", "evaluation"):
        with pytest.raises(runner.ApprovalRequired, match="approval"):
            runner.require_approval(name, False)
        runner.require_approval(name, True)
    runner.require_approval("probe", False)


def test_estimate_is_exact_and_sees_cached_first_steps(mind_frames, tmp_path):
    eps, _, _ = _setup(mind_frames)
    teacher = api.Teacher(
        FakeClient(), "cat", cache=api.ResponseCache(tmp_path / "c.sqlite"), sleep=lambda s: None
    )
    est = runner.estimate(eps, ["V1", "V2"], teacher=teacher)
    expected = sum(requests_needed(ep.n, E.episode_k(ep), "V1") for ep in eps)
    assert est["V1"]["requests"] == expected == est["V2"]["requests"]
    assert est["total_requests"] == 2 * expected
    assert est["V1"]["first_step_requests_cached"] == 0
    _rows, facts = runner.run_variant(teacher, eps, "V1", workers=3)
    assert facts["requests"] == expected == facts["teacher"]["api_calls"]
    again = runner.estimate(eps, ["V1"], teacher=teacher)
    assert again["V1"]["first_step_requests_cached"] == len(eps)


def test_run_variant_rows_and_resume_from_cache(mind_frames, tmp_path):
    eps, _, _ = _setup(mind_frames)
    client = FakeClient()
    cache = api.ResponseCache(tmp_path / "c.sqlite")
    teacher = api.Teacher(client, "cat", cache=cache, sleep=lambda s: None)
    seen = []
    rows, facts = runner.run_variant(
        teacher, eps, "V2", workers=4, progress=lambda d, t: seen.append((d, t))
    )
    assert seen[-1] == (len(eps), len(eps))
    assert len(rows) == sum(ep.n for ep in eps) and facts["records"] == len(eps)
    assert set(rows.groupby("record_id")["predicted_attention"].sum()) <= {
        E.episode_k(ep) for ep in eps
    }
    calls = len(client.calls)
    teacher2 = api.Teacher(client, "cat", cache=cache, sleep=lambda s: None)
    rows2, facts2 = runner.run_variant(teacher2, eps, "V2", workers=2)
    assert len(client.calls) == calls and facts2["teacher"]["cache_hits"] == facts["requests"]
    pd.testing.assert_frame_equal(
        rows.sort_values(["record_id", "news_id"]).reset_index(drop=True),
        rows2.sort_values(["record_id", "news_id"]).reset_index(drop=True),
    )
    path = runner.predictions_path(tmp_path, "selection", "V2", False)
    runner.save_predictions(rows, facts, path)
    assert pd.read_parquet(path).shape == rows.shape and path.with_suffix(".json").is_file()


def test_choose_variant_picks_best_auc_and_prefers_simpler_on_ties(mind_frames):
    _eps, dec, recs = _setup(mind_frames)
    sel = dec.copy()
    perfect = sel.assign(
        confidence=sel["attended"].astype(float) + 0.01, predicted_attention=sel["attended"]
    )
    reverse = sel.assign(confidence=1.0 - sel["attended"], predicted_attention=0)
    keep = ["record_set", "record_id", "news_id", "confidence", "predicted_attention"]
    best, table = runner.choose_variant(
        {"V1": reverse[keep], "V2": perfect[keep]}, dec, recs, n_boot=50
    )
    assert best == "V2" and table["V2"]["value"] == 1.0 and table["V1"]["value"] == 0.0
    tie, _ = runner.choose_variant(
        {"V2": perfect[keep], "V1o": perfect[keep]}, dec, recs, n_boot=50
    )
    assert tie == "V1o"


def test_with_decisions_requires_full_coverage(mind_frames):
    _, dec, _ = _setup(mind_frames)
    pred = dec[["record_set", "record_id", "news_id"]].assign(confidence=0.5).iloc[:-1]
    extra = pd.concat([pred, pred.iloc[:1].assign(news_id="N_unknown")])
    with pytest.raises(ValueError):
        runner.with_decisions(extra, dec)
