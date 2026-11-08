"""Baselines without BehaviorGPT (plans/R1.md 8). Each gives every presented observation a
score (higher = more likely attended); decisions use the same k rule as BehaviorGPT.

- random: a seeded hash of the record and the observation.
- popularity: the observation's click rate in all impressions (both files) strictly
  before the record's time, shrunk towards the overall click rate before that time:
  (clicks + A * rate) / (shows + A), with A = 10 pseudo-shows. Only earlier impressions,
  so nothing of the record itself or later.
- recency: how recently the observation first appeared in any impression strictly before
  the record (never shown before = brand new = most recent).
- content similarity: TF-IDF cosine between the observation's title and the mean TF-IDF
  vector of the titles the person attended before (History clicks and earlier impression
  clicks of the episode); 0 for a person with no earlier clicks.

None of them reads a real decision of the episode being scored.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from significance import episodes as E

POPULARITY_PSEUDO_SHOWS = 10.0
BASELINES = ("random", "popularity", "recency", "content")
_KEY_SCALE = 10**8  # seconds per news code in the combined sort key (MIND spans ~2 months)


def random_scores(frame: pd.DataFrame, seed: int = 0) -> np.ndarray:
    out = np.empty(len(frame))
    for i, (rs, ri, n) in enumerate(
        zip(frame["record_set"], frame["record_id"], frame["news_id"], strict=True)
    ):
        h = hashlib.sha256(f"random:{seed}:{rs}:{ri}:{n}".encode()).hexdigest()
        out[i] = int(h[:12], 16) / 16**12
    return out


class ExposureIndex:
    """When each news article was shown and clicked, over all impressions of both files,
    to answer 'before time t' questions without looking at t or later."""

    def __init__(self, impressions: Sequence[pd.DataFrame]):
        both = pd.concat([imp[["time", "presented", "clicked"]] for imp in impressions])
        self.t0 = both["time"].min()
        sec = ((both["time"] - self.t0).dt.total_seconds()).astype("int64").to_numpy()
        lengths = both["presented"].map(len).to_numpy()
        news = np.concatenate([np.asarray(p, dtype=object) for p in both["presented"]])
        clicks = np.concatenate([np.asarray(c, dtype=np.int64) for c in both["clicked"]])
        times = np.repeat(sec, lengths)
        codes, self.news = pd.factorize(pd.Series(news))
        self.code_of = {n: i for i, n in enumerate(self.news)}
        key = codes.astype(np.int64) * _KEY_SCALE + times
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.cum_clicks = np.concatenate([[0], np.cumsum(clicks[order])])
        self.first_index = np.searchsorted(self.key, np.arange(len(self.news)) * _KEY_SCALE)
        self.first_time = times[order][self.first_index]
        by_time = np.argsort(times, kind="stable")
        self.all_times = times[by_time]
        self.all_cum_clicks = np.concatenate([[0], np.cumsum(clicks[by_time])])

    def seconds(self, times: pd.Series) -> np.ndarray:
        return ((pd.Series(times) - self.t0).dt.total_seconds()).astype("int64").to_numpy()

    def before(self, news_ids: Sequence[str], times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """(shows, clicks) of each news article in impressions strictly before its time."""
        t = self.seconds(times)
        codes = np.array([self.code_of.get(n, -1) for n in news_ids])
        shows = np.zeros(len(codes), dtype=np.int64)
        clicks = np.zeros(len(codes), dtype=np.int64)
        known = codes >= 0
        qkey = codes[known].astype(np.int64) * _KEY_SCALE + t[known]
        idx = np.searchsorted(self.key, qkey, side="left")
        start = self.first_index[codes[known]]
        shows[known] = idx - start
        clicks[known] = self.cum_clicks[idx] - self.cum_clicks[start]
        return shows, clicks

    def overall_rate_before(self, times: pd.Series) -> np.ndarray:
        t = self.seconds(times)
        idx = np.searchsorted(self.all_times, t, side="left")
        return np.where(idx > 0, self.all_cum_clicks[idx] / np.maximum(idx, 1), 0.0)

    def age_before(self, news_ids: Sequence[str], times: pd.Series) -> np.ndarray:
        """Seconds since first shown, counting impressions strictly before; 0 if never."""
        t = self.seconds(times)
        codes = np.array([self.code_of.get(n, -1) for n in news_ids])
        first = np.where(codes >= 0, self.first_time[np.maximum(codes, 0)], np.iinfo(np.int64).max)
        return np.where(first < t, t - first, 0).astype(float)


def _record_times(frame: pd.DataFrame, records: pd.DataFrame) -> pd.Series:
    times = records.set_index(["record_set", "record_id"])["time"]
    return pd.Series(
        times.loc[list(zip(frame["record_set"], frame["record_id"], strict=True))].to_numpy()
    )


def popularity_scores(frame: pd.DataFrame, records: pd.DataFrame, index: ExposureIndex):
    t = _record_times(frame, records)
    shows, clicks = index.before(frame["news_id"].tolist(), t)
    prior = index.overall_rate_before(t)
    a = POPULARITY_PSEUDO_SHOWS
    return (clicks + a * prior) / (shows + a)


def recency_scores(frame: pd.DataFrame, records: pd.DataFrame, index: ExposureIndex):
    t = _record_times(frame, records)
    return -index.age_before(frame["news_id"].tolist(), t)


def attended_before(episode: E.Episode) -> list[str]:
    return [n for p in episode.previous for n in p.attended]


class ContentModel:
    """TF-IDF of news titles (fitted on titles only, no decisions)."""

    def __init__(self, news: pd.DataFrame):
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
        self.matrix = normalize(self.vectorizer.fit_transform(news["title"]))
        self.row = {n: i for i, n in enumerate(news["news_id"])}

    def scores(self, frame: pd.DataFrame, episodes: Sequence[E.Episode]) -> np.ndarray:
        by_key = {(ep.record_set, ep.record_id): ep for ep in episodes}
        out = np.zeros(len(frame))
        positions = frame.groupby(["record_set", "record_id"]).indices
        for key, pos in positions.items():
            past = [self.row[n] for n in attended_before(by_key[key]) if n in self.row]
            if not past:
                continue
            profile = normalize(np.asarray(self.matrix[past].mean(axis=0)))
            cand = [self.row.get(n, -1) for n in frame["news_id"].iloc[pos]]
            ok = [i for i, c in enumerate(cand) if c >= 0]
            sims = self.matrix[[cand[i] for i in ok]] @ profile.T
            out[pos[ok]] = np.asarray(sims).ravel()
        return out


def top_k_decisions(frame: pd.DataFrame, scores: np.ndarray, k_by_record: dict) -> np.ndarray:
    """1 for the k highest-scoring observations of each record (the same k rule as
    BehaviorGPT), ties broken by the record/item hash."""
    from significance.evaluate import strict_scores

    ranked = strict_scores(frame.assign(_b=scores).reset_index(drop=True), "_b")
    rank = (-ranked).astype(int).to_numpy()
    k = np.array(
        [k_by_record[key] for key in zip(frame["record_set"], frame["record_id"], strict=True)]
    )
    return (rank <= k).astype(int)


def all_baseline_scores(
    frame: pd.DataFrame,
    records: pd.DataFrame,
    episodes: Sequence[E.Episode],
    index: ExposureIndex,
    content: ContentModel,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """The four baseline scores for every observation of `frame`."""
    frame = frame.reset_index(drop=True)
    return pd.DataFrame(
        {
            "random": random_scores(frame, seed),
            "popularity": popularity_scores(frame, records, index),
            "recency": recency_scores(frame, records, index),
            "content": content.scores(frame, episodes),
        }
    )
