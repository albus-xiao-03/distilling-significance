"""Assess predictors of the attention decisions on a record set (plans/R1.md 8, E1-E3).

A predictor gives every presented observation a score (higher = attended earlier) and a
predicted decision. `assess` compares both with the real decisions; `compare` gives a
paired difference between two predictors; `by_stratum` splits records into those whose
person has earlier impressions (presented sets) and those with only History clicks or
nothing, because only the former can show what the observation sequence adds (V2) and the
two strata differ strongly between the selection and evaluation records (Phase 1).

Everything returned is aggregate: means, pooled rates and their intervals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from significance import episodes as E
from significance import evaluate as M


def predictor_frame(
    decisions: pd.DataFrame, score: Sequence[float], predicted: Sequence[int]
) -> pd.DataFrame:
    """The decisions table with a predictor's score and predicted decision."""
    frame = decisions.reset_index(drop=True).copy()
    frame["score"] = np.asarray(score, dtype=float)
    frame["predicted_attention"] = np.asarray(predicted, dtype=int)
    return frame


def bgpt_frame(predictions: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """BehaviorGPT predictions aligned to the decisions table (score = confidence)."""
    keys = ["record_set", "record_id", "news_id"]
    merged = decisions.reset_index(drop=True).merge(
        predictions[[*keys, "confidence", "predicted_attention"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged["confidence"].isna().any():
        raise ValueError("predictions do not cover every presented observation")
    return merged.rename(columns={"confidence": "score"})


def assess(
    frame: pd.DataFrame, persons: pd.Series, *, n_boot: int = 1000, seed: int = 0
) -> tuple[dict, pd.DataFrame]:
    """E1 ranking metrics, E2 decision metrics, the k error and the oracle-k precision,
    each with a 95 % person-bootstrap interval. `persons` maps record_id to person.
    Returns (aggregates, per-record ranking table for paired comparisons)."""
    per = M.ranking_table(frame, "score")
    who = persons.loc[per["record_id"]].tolist()
    out: dict = {
        "E1": {
            m: M.mean_with_ci(per[m].tolist(), who, n_boot=n_boot, seed=seed).as_dict()
            for m in M.RANKING_METRICS
        }
    }
    dec = M.decision_table(frame)
    who_dec = persons.loc[dec["record_id"]].tolist()
    out["E2"] = {
        name: est.as_dict()
        for name, est in M.decision_metrics_with_ci(dec, who_dec, n_boot=n_boot, seed=seed).items()
    }
    err = (dec["k"] - dec["true_clicks"]).abs()
    out["k"] = {
        "mean_k": round(float(dec["k"].mean()), 3),
        "mean_true_clicks": round(float(dec["true_clicks"].mean()), 3),
        "mean_abs_error": M.mean_with_ci(err.tolist(), who_dec, n_boot=n_boot, seed=seed).as_dict(),
        "share_exact": round(float((err == 0).mean()), 4),
    }
    oracle = M.oracle_k_table(frame, "score")
    who_or = persons.loc[oracle["record_id"]].tolist()
    reps, n_persons = M.bootstrap_ratio(
        [oracle["oracle_hits"].to_numpy()],
        [oracle["true_clicks"].to_numpy()],
        who_or,
        lambda h, c: h / c,
        n_boot=n_boot,
        seed=seed,
    )
    point = oracle["oracle_hits"].sum() / oracle["true_clicks"].sum()
    out["oracle_k_precision"] = M.Estimate(
        float(point), *np.percentile(reps, [2.5, 97.5]), len(oracle), n_persons
    ).as_dict()
    return out, per


def compare(
    per_a: pd.DataFrame,
    per_b: pd.DataFrame,
    persons: pd.Series,
    metric: str = "auc",
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired difference a - b of a per-record metric, with a person-bootstrap interval."""
    both = per_a[["record_id", metric]].merge(
        per_b[["record_id", metric]], on="record_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    a = [None if pd.isna(x) else float(x) for x in both[f"{metric}_a"]]
    b = [None if pd.isna(x) else float(x) for x in both[f"{metric}_b"]]
    who = persons.loc[both["record_id"]].tolist()
    return M.paired_diff_with_ci(a, b, who, n_boot=n_boot, seed=seed).as_dict()


def strata(episodes: Sequence[E.Episode]) -> pd.Series:
    """record_id -> 'earlier impressions' | 'history clicks only' | 'no previous decisions'."""

    def label(ep: E.Episode) -> str:
        if ep.has_earlier_impressions:
            return "earlier impressions"
        return "history clicks only" if ep.previous else "no previous decisions"

    return pd.Series({ep.record_id: label(ep) for ep in episodes})


def by_stratum(
    per: pd.DataFrame,
    labels: pd.Series,
    persons: pd.Series,
    metric: str = "auc",
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    out = {}
    for name in sorted(labels.unique()):
        part = per[per["record_id"].map(labels) == name]
        if len(part):
            out[name] = M.mean_with_ci(
                part[metric].tolist(),
                persons.loc[part["record_id"]].tolist(),
                n_boot=n_boot,
                seed=seed,
            ).as_dict()
    return out


def without_seen_before(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame without observations seen in an earlier episode (re-presented ones,
    plans/R1.md L7), for a sensitivity check of E1."""
    return frame[~frame["seen_before"]].reset_index(drop=True)
