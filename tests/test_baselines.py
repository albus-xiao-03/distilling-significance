"""Baselines: exposure index strictly before t, smoothing, content profile, no leakage."""

import numpy as np
import pandas as pd
import pytest

from significance import baselines as B
from significance import episodes as E


def _imps(rows):
    """rows: (time string, presented tuple, clicked tuple)"""
    return pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "presented": [r[1] for r in rows],
            "clicked": [r[2] for r in rows],
        }
    )


TRAIN = _imps(
    [
        ("2019-11-09 10:00", ("X", "Y"), (1, 0)),
        ("2019-11-10 10:00", ("X", "Z"), (0, 1)),
    ]
)
DEV = _imps([("2019-11-15 09:00", ("X", "Y", "Z"), (1, 0, 0))])


def test_exposure_counts_only_strictly_earlier_impressions():
    idx = B.ExposureIndex([TRAIN, DEV])
    t = pd.Series(pd.to_datetime(["2019-11-10 10:00", "2019-11-15 09:00", "2019-11-16 00:00"]))
    shows, clicks = idx.before(["X", "X", "X"], t)
    assert shows.tolist() == [1, 2, 3] and clicks.tolist() == [1, 1, 2]
    s, c = idx.before(["Q"], t.iloc[:1])
    assert s.tolist() == [0] and c.tolist() == [0]
    rate = idx.overall_rate_before(
        pd.Series(pd.to_datetime(["2019-11-09 10:00", "2019-11-12 00:00"]))
    )
    assert rate.tolist() == [0.0, 2 / 4]


def test_age_before_and_recency():
    idx = B.ExposureIndex([TRAIN, DEV])
    t = pd.Series(pd.to_datetime(["2019-11-09 10:00", "2019-11-10 10:00", "2019-11-10 10:00"]))
    age = idx.age_before(["X", "X", "Z"], t)
    assert age.tolist() == [0.0, 86400.0, 0.0]  # Z first shown exactly at t -> not before


def _frame_and_records():
    frame = pd.DataFrame(
        {
            "record_set": ["evaluation"] * 3,
            "record_id": [0, 0, 0],
            "news_id": ["X", "Y", "Z"],
            "attended": [1, 0, 0],
        }
    )
    records = pd.DataFrame(
        {
            "record_set": ["evaluation"],
            "record_id": [0],
            "time": pd.to_datetime(["2019-11-15 09:00"]),
            "user_id": ["U1"],
        }
    )
    return frame, records


def test_popularity_is_smoothed_and_ignores_the_record_itself():
    frame, records = _frame_and_records()
    idx = B.ExposureIndex([TRAIN, DEV])
    pop = B.popularity_scores(frame, records, idx)
    prior = 2 / 4
    a = B.POPULARITY_PSEUDO_SHOWS
    assert pop.tolist() == pytest.approx(
        [(1 + a * prior) / (2 + a), (0 + a * prior) / (1 + a), (1 + a * prior) / (1 + a)]
    )
    flipped = _imps([("2019-11-15 09:00", ("X", "Y", "Z"), (0, 1, 1))])
    pop2 = B.popularity_scores(frame, records, B.ExposureIndex([TRAIN, flipped]))
    assert pop2.tolist() == pytest.approx(pop.tolist())


def test_random_scores_are_seeded():
    frame, _ = _frame_and_records()
    a, b = B.random_scores(frame, 1), B.random_scores(frame, 1)
    assert a.tolist() == b.tolist() and a.tolist() != B.random_scores(frame, 2).tolist()
    assert ((a >= 0) & (a < 1)).all()


def test_content_model_prefers_titles_like_past_clicks():
    news = pd.DataFrame(
        {
            "news_id": ["G1", "G2", "G3", "P1", "P2", "P3"],
            "title": [
                "golf masters tournament leader",
                "golf open tournament champion",
                "golf tournament final round",
                "senate election vote count",
                "election senate debate",
                "vote senate bill",
            ],
        }
    )
    model = B.ContentModel(news)
    ep = E.Episode("evaluation", 0, (E.EarlierEpisode(None, ("G1", "G2")),), ("G3", "P3"), 0, 0)
    frame = pd.DataFrame(
        {"record_set": ["evaluation"] * 2, "record_id": [0, 0], "news_id": ["G3", "P3"]}
    )
    s = model.scores(frame, [ep])
    assert s[0] > s[1]
    cold = E.Episode("evaluation", 0, (), ("G3", "P3"), 0, 0)
    assert model.scores(frame, [cold]).tolist() == [0.0, 0.0]


def test_top_k_decisions_and_no_label_reads():
    frame, records = _frame_and_records()
    scores = np.array([0.1, 0.9, 0.5])
    assert B.top_k_decisions(frame, scores, {("evaluation", 0): 2}).tolist() == [0, 1, 1]
    idx = B.ExposureIndex([TRAIN, DEV])
    news = pd.DataFrame(
        {"news_id": ["X", "Y", "Z"], "title": ["alpha beta", "alpha gamma", "beta gamma"]}
    )
    ep = E.Episode("evaluation", 0, (E.EarlierEpisode(None, ("X",)),), ("X", "Y", "Z"), 0, 0)
    content = B.ContentModel(news)
    s1 = B.all_baseline_scores(frame, records, [ep], idx, content)
    s2 = B.all_baseline_scores(frame.assign(attended=[0, 1, 1]), records, [ep], idx, content)
    pd.testing.assert_frame_equal(s1, s2)
    assert list(s1.columns) == list(B.BASELINES)
