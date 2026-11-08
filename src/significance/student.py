"""E4, distillation: does a student trained on the pseudo-labels learn the real decisions?
(plans/R1.md 8)

The student is a logistic regression on features that do not come from BehaviorGPT:

- the title's TF-IDF (fitted on all titles, no decisions);
- the observation's category and subcategory (one-hot);
- the person's past attention share in that category: the share of the observations the
  person attended before (History clicks and earlier impression clicks) that are in it;
- the observation's earlier click rate (the popularity baseline: impressions strictly
  before the record);
- history length (log of 1 + earlier attended observations) and episode size (log of n).

It is trained on 80 % of the records, split by person, with one label source at a time:
BehaviorGPT's pseudo-labels, the real decisions (upper bound), popularity labels and
random labels (each top-k with the same k rule). It is always evaluated on the REAL
decisions of the other 20 %. Recovery = (AUC_pseudo - 0.5) / (AUC_real - 0.5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from significance import baselines as B
from significance import episodes as E
from significance import evaluate as M

NUMERIC = ("past_category_share", "earlier_click_rate", "log_history_length", "log_episode_size")


def numeric_features(
    frame: pd.DataFrame,
    episodes: Sequence[E.Episode],
    records: pd.DataFrame,
    news: pd.DataFrame,
    index: B.ExposureIndex,
) -> pd.DataFrame:
    """The four numeric features for every observation of `frame` (positional order)."""
    frame = frame.reset_index(drop=True)
    category = dict(zip(news["news_id"], news["category"], strict=True))
    by_key = {(ep.record_set, ep.record_id): ep for ep in episodes}
    share = np.zeros(len(frame))
    hist_len = np.zeros(len(frame))
    size = np.zeros(len(frame))
    for key, pos in frame.groupby(["record_set", "record_id"]).indices.items():
        ep = by_key[key]
        past = [category.get(n) for n in B.attended_before(ep)]
        counts = pd.Series(past, dtype=object).value_counts()
        cats = frame["news_id"].iloc[pos].map(category)
        share[pos] = (cats.map(counts).fillna(0) / max(len(past), 1)).to_numpy()
        hist_len[pos] = np.log1p(len(past))
        size[pos] = np.log1p(ep.n)
    return pd.DataFrame(
        {
            "past_category_share": share,
            "earlier_click_rate": B.popularity_scores(frame, records, index),
            "log_history_length": hist_len,
            "log_episode_size": size,
        }
    )


class Features:
    """Builds the student's design matrix; text and category encoders are fitted on all
    news (no decisions), the numeric scaler on the training rows only."""

    def __init__(self, news: pd.DataFrame, content: B.ContentModel):
        self.content = content
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.encoder.fit(news[["category", "subcategory"]].to_numpy())
        self.cats = news.set_index("news_id")[["category", "subcategory"]]
        self.scaler = StandardScaler()

    def matrix(self, frame: pd.DataFrame, numeric: pd.DataFrame, fit_rows=None) -> sp.csr_matrix:
        rows = [self.content.row[n] for n in frame["news_id"]]
        text = self.content.matrix[rows]
        onehot = self.encoder.transform(self.cats.loc[frame["news_id"]].to_numpy())
        values = numeric[list(NUMERIC)].to_numpy()
        if fit_rows is not None:
            self.scaler.fit(values[fit_rows])
        return sp.hstack([text, onehot, sp.csr_matrix(self.scaler.transform(values))]).tocsr()


def split_by_person(records: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """(train record ids, test record ids): no person is in both."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(records, groups=records["user_id"]))
    return (
        set(records["record_id"].iloc[train_idx].tolist()),
        set(records["record_id"].iloc[test_idx].tolist()),
    )


def fit_predict(x_train, y_train, x_test, seed: int = 0) -> np.ndarray:
    if len(set(np.asarray(y_train).tolist())) < 2:
        raise ValueError("the training labels hold one class only")
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def distil(
    frame: pd.DataFrame,
    numeric: pd.DataFrame,
    features: Features,
    persons: pd.Series,
    labels: dict[str, np.ndarray],
    train_ids: set,
    test_ids: set,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Train one student per label source; evaluate each on the test records' real
    decisions (per-record AUC with person-bootstrap CI, and pooled AUC)."""
    frame = frame.reset_index(drop=True)
    is_train = frame["record_id"].isin(train_ids).to_numpy()
    is_test = frame["record_id"].isin(test_ids).to_numpy()
    if (is_train & is_test).any():
        raise ValueError("a record is in both the training and the test split")
    x = features.matrix(frame, numeric, fit_rows=is_train)
    test = frame[is_test].reset_index(drop=True)
    who = None
    out: dict = {"train_records": len(train_ids), "test_records": len(test_ids), "students": {}}
    per_record = {}
    for name, y in labels.items():
        prob = fit_predict(x[is_train], np.asarray(y)[is_train], x[is_test], seed=seed)
        per = M.ranking_table(test.assign(score=prob), "score")
        who = persons.loc[per["record_id"]].tolist()
        est = M.mean_with_ci(per["auc"].tolist(), who, n_boot=n_boot, seed=seed)
        pooled = M.auc(test["attended"].to_numpy(), prob)
        out["students"][name] = {"auc": est.as_dict(), "pooled_auc": round(float(pooled), 4)}
        per_record[name] = per
    if "pseudo" in per_record and "real" in per_record:
        out["recovery"] = recovery_with_ci(
            per_record["pseudo"], per_record["real"], persons, n_boot=n_boot, seed=seed
        )
    for name in ("popularity", "random"):
        if name in per_record and "real" in per_record:
            out[f"recovery_{name}"] = recovery_with_ci(
                per_record[name], per_record["real"], persons, n_boot=n_boot, seed=seed
            )
    return out


def recovery_with_ci(
    per_a: pd.DataFrame,
    per_real: pd.DataFrame,
    persons: pd.Series,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """(mean AUC_a - 0.5) / (mean AUC_real - 0.5), with a paired person bootstrap."""
    both = (
        per_a[["record_id", "auc"]]
        .merge(per_real[["record_id", "auc"]], on="record_id", suffixes=("_a", "_r"))
        .dropna()
    )
    who = persons.loc[both["record_id"]].tolist()
    a, r = both["auc_a"].to_numpy(), both["auc_r"].to_numpy()
    reps, _ = M.bootstrap_ratio(
        [a, r],
        [np.ones_like(a)],
        who,
        lambda sa, sr, c: (sa / c - 0.5) / (sr / c - 0.5),
        n_boot=n_boot,
        seed=seed,
    )
    point = (a.mean() - 0.5) / (r.mean() - 0.5)
    return {
        "value": round(float(point), 4),
        "ci95": [round(float(x), 4) for x in np.percentile(reps, [2.5, 97.5])],
    }
