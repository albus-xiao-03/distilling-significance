"""Evaluation against the real human decisions (plans/R1.md 8).

- Ranking metrics per record (E1): AUC, MRR, nDCG@5 and nDCG@10, with MIND's own formulas
  (gains 2^y - 1, log2 discounts); a record's ranking is a strict order of its presented
  observations, ties broken by a hash of the record and the item.
- Decision metrics (E2): precision, recall, F1, accuracy and Cohen's kappa of the
  predicted decisions, pooled over observations; the error in k; the oracle-k diagnostic
  (top true-click-count by rank), which separates ranking errors from count errors.
- Uncertainty: 95 % bootstrap intervals over persons (all records of a resampled person go
  together), and paired bootstrap intervals for differences between predictors (E3).

Only aggregates leave this module's callers: no per-record values are ever reported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from significance.teacher.decode import tiebreak

RANKING_METRICS = ("auc", "mrr", "ndcg5", "ndcg10")


# --- per-record ranking metrics (E1) --------------------------------------------------


def auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """P(score of an attended observation > score of a not-attended one), ties 1/2; None
    when the record has only one class."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def _sorted_labels(labels, scores) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return np.asarray(labels, dtype=float)[order]


def mrr(labels: Sequence[int], scores: Sequence[float]) -> float:
    """MIND's MRR: sum over attended observations of 1/rank, divided by their number
    (NaN, and so left out of means, when nothing was attended)."""
    y = _sorted_labels(labels, scores)
    if not y.sum():
        return float("nan")
    return float(np.sum(y / (np.arange(len(y)) + 1)) / np.sum(y))


def _dcg(y_sorted: np.ndarray, k: int) -> float:
    y = y_sorted[:k]
    return float(np.sum((2**y - 1) / np.log2(np.arange(len(y)) + 2)))


def ndcg(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """MIND's nDCG@k (NaN when nothing was attended)."""
    best = _dcg(np.sort(np.asarray(labels, dtype=float))[::-1], k)
    return _dcg(_sorted_labels(labels, scores), k) / best if best > 0 else float("nan")


def strict_scores(frame: pd.DataFrame, score_col: str) -> pd.Series:
    """Scores turned into a strict order within each record (higher = earlier), ties
    broken by the record/item hash, so every metric sees the same ranking."""
    tb = [
        tiebreak(rs, int(ri), n)
        for rs, ri, n in zip(frame["record_set"], frame["record_id"], frame["news_id"], strict=True)
    ]
    keyed = frame.assign(_s=frame[score_col].astype(float).fillna(-np.inf), _t=tb)
    keyed = keyed.sort_values(
        ["record_set", "record_id", "_s", "_t"], ascending=[True, True, False, True]
    )
    rank = keyed.groupby(["record_set", "record_id"]).cumcount() + 1
    return (-rank).reindex(frame.index).astype(float)


def ranking_table(frame: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Per-record E1 metrics. `frame` holds record_set, record_id, news_id, attended and
    `score_col` (higher = attended earlier)."""
    strict = strict_scores(frame, score_col)
    rows = []
    for (rs, ri), idx in frame.groupby(["record_set", "record_id"]).groups.items():
        y = frame.loc[idx, "attended"].to_numpy()
        s = strict.loc[idx].to_numpy()
        rows.append(
            {
                "record_set": rs,
                "record_id": int(ri),
                "auc": auc(y, s),
                "mrr": mrr(y, s),
                "ndcg5": ndcg(y, s, 5),
                "ndcg10": ndcg(y, s, 10),
            }
        )
    return pd.DataFrame(rows)


# --- decision metrics (E2) ------------------------------------------------------------


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(((t == 1) & (p == 1)).sum()),
        "fp": int(((t == 0) & (p == 1)).sum()),
        "fn": int(((t == 1) & (p == 0)).sum()),
        "tn": int(((t == 0) & (p == 0)).sum()),
    }


def from_confusion(tp, fp, fn, tn) -> dict[str, float]:
    """Precision, recall, F1, accuracy and Cohen's kappa from counts (arrays allowed)."""
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        accuracy = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
        kappa = np.where(pe < 1, (accuracy - pe) / (1 - pe), 0.0)
    out = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "kappa": kappa}
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}


def decision_table(frame: pd.DataFrame, pred_col: str = "predicted_attention") -> pd.DataFrame:
    """Per-record confusion counts, true clicks and predicted k."""
    g = frame.assign(
        tp=(frame["attended"] == 1) & (frame[pred_col] == 1),
        fp=(frame["attended"] == 0) & (frame[pred_col] == 1),
        fn=(frame["attended"] == 1) & (frame[pred_col] == 0),
        tn=(frame["attended"] == 0) & (frame[pred_col] == 0),
    ).groupby(["record_set", "record_id"])
    out = g[["tp", "fp", "fn", "tn"]].sum().astype(int)
    out["true_clicks"] = g["attended"].sum().astype(int)
    out["k"] = g[pred_col].sum().astype(int)
    return out.reset_index()


def oracle_k_table(frame: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Per record, the precision of the top-(true clicks) observations by rank: the
    decisions BehaviorGPT would make if it were told how many to attend (diagnostic)."""
    strict = strict_scores(frame, score_col)
    keyed = frame.assign(_s=strict)
    rows = []
    for (rs, ri), part in keyed.groupby(["record_set", "record_id"]):
        m = int(part["attended"].sum())
        top = part.nlargest(m, "_s")
        rows.append(
            {
                "record_set": rs,
                "record_id": int(ri),
                "oracle_hits": int(top["attended"].sum()),
                "true_clicks": m,
            }
        )
    return pd.DataFrame(rows)


# --- bootstrap over persons -----------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n_records: int
    n_persons: int

    def as_dict(self, digits: int = 4) -> dict:
        return {
            "value": round(self.value, digits),
            "ci95": [round(self.lo, digits), round(self.hi, digits)],
            "records": self.n_records,
            "persons": self.n_persons,
        }


def _person_weights(persons: Sequence[str], n_boot: int, seed: int):
    codes, uniques = pd.factorize(pd.Series(list(persons)))
    p = len(uniques)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(p, np.full(p, 1.0 / p), size=n_boot)  # (B, P)
    return codes, p, counts


def bootstrap_ratio(
    numerators: Sequence[np.ndarray],
    denominators: Sequence[np.ndarray] | None,
    persons: Sequence[str],
    stat: Callable[..., np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Bootstrap replicates of `stat(*person_sums)`: per-record arrays are summed per
    person, persons are resampled with replacement, and `stat` gets the resampled sums."""
    codes, p, counts = _person_weights(persons, n_boot, seed)
    arrays = list(numerators) + list(denominators or [])
    sums = [np.bincount(codes, weights=np.asarray(a, dtype=float), minlength=p) for a in arrays]
    replicated = [counts @ s for s in sums]
    return stat(*replicated), p


def mean_with_ci(
    values: Sequence[float | None], persons: Sequence[str], *, n_boot=1000, seed=0
) -> Estimate:
    """Mean over records with a 95 % bootstrap interval over persons (None values, e.g. an
    undefined AUC, are left out)."""
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    keep = ~np.isnan(v)
    v, per = v[keep], [p for p, k in zip(persons, keep, strict=True) if k]
    reps, n_persons = bootstrap_ratio(
        [v], [np.ones_like(v)], per, lambda s, c: s / c, n_boot=n_boot, seed=seed
    )
    return Estimate(float(v.mean()), *np.percentile(reps, [2.5, 97.5]), len(v), n_persons)


def paired_diff_with_ci(
    a: Sequence[float | None],
    b: Sequence[float | None],
    persons: Sequence[str],
    *,
    n_boot=1000,
    seed=0,
) -> Estimate:
    """Mean of a - b over records where both are defined, with a paired person bootstrap."""
    d = [None if x is None or y is None else x - y for x, y in zip(a, b, strict=True)]
    return mean_with_ci(d, persons, n_boot=n_boot, seed=seed)


def decision_metrics_with_ci(
    table: pd.DataFrame, persons: Sequence[str], *, n_boot=1000, seed=0
) -> dict[str, Estimate]:
    """Pooled precision, recall, F1, accuracy and kappa with person-bootstrap intervals."""
    cols = [table[c].to_numpy() for c in ("tp", "fp", "fn", "tn")]
    point = from_confusion(*(c.sum() for c in cols))
    out = {}
    for name in point:
        reps, n_persons = bootstrap_ratio(
            cols,
            None,
            persons,
            lambda *s, name=name: from_confusion(*s)[name],
            n_boot=n_boot,
            seed=seed,
        )
        out[name] = Estimate(point[name], *np.percentile(reps, [2.5, 97.5]), len(table), n_persons)
    return out
