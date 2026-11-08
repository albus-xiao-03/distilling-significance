"""Evaluation metrics against known values and sklearn; person bootstrap."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from significance import evaluate as M


def test_auc_matches_hand_value_and_sklearn():
    y, s = [1, 0, 0, 1], [0.9, 0.1, 0.8, 0.3]
    assert M.auc(y, s) == 0.75 == roc_auc_score(y, s)
    assert M.auc([1, 0, 1], [0.5, 0.5, 0.2]) == pytest.approx(
        roc_auc_score([1, 0, 1], [0.5, 0.5, 0.2])
    )
    assert M.auc([1, 1], [0.1, 0.2]) is None


def test_mrr_and_ndcg_follow_mind_formulas():
    y, s = [0, 1, 0, 1], [0.4, 0.3, 0.2, 0.1]
    assert M.mrr(y, s) == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert M.ndcg(y, s, 2) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert M.ndcg([1, 0], [0.9, 0.1], 10) == 1.0
    assert np.isnan(M.mrr([0, 0], [0.2, 0.1])) and np.isnan(M.ndcg([0, 0], [0.2, 0.1], 5))
    est = M.mean_with_ci([0.5, float("nan"), 0.7], ["u1", "u2", "u3"], n_boot=20)
    assert est.value == pytest.approx(0.6) and est.n_records == 2


def _frame():
    return pd.DataFrame(
        {
            "record_set": ["s"] * 6,
            "record_id": [0, 0, 0, 1, 1, 1],
            "news_id": ["a", "b", "c", "a", "b", "c"],
            "attended": [1, 0, 0, 0, 0, 1],
            "score": [0.9, 0.5, 0.1, 0.9, 0.5, 0.1],
            "predicted_attention": [1, 0, 0, 1, 0, 0],
        }
    )


def test_ranking_table_per_record():
    t = M.ranking_table(_frame()).set_index("record_id")
    assert t.loc[0, "auc"] == 1.0 and t.loc[1, "auc"] == 0.0
    assert t.loc[0, "mrr"] == 1.0 and t.loc[1, "mrr"] == pytest.approx(1 / 3)


def test_strict_scores_break_ties_by_hash_not_input_order():
    f = _frame().assign(score=0.5)
    s1 = M.strict_scores(f, "score")
    s2 = M.strict_scores(f.iloc[::-1].reset_index(drop=True), "score")
    m1 = dict(zip(zip(f["record_id"], f["news_id"], strict=True), s1, strict=True))
    f2 = f.iloc[::-1].reset_index(drop=True)
    m2 = dict(zip(zip(f2["record_id"], f2["news_id"], strict=True), s2, strict=True))
    assert m1 == m2 and sorted(set(s1)) == [-3.0, -2.0, -1.0]


def test_decision_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    t, p = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    c = M.confusion(t, p)
    m = M.from_confusion(**c)
    assert m["f1"] == pytest.approx(f1_score(t, p))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(t, p))
    assert m["accuracy"] == pytest.approx((t == p).mean())


def test_decision_and_oracle_tables():
    f = _frame()
    d = M.decision_table(f).set_index("record_id")
    assert d.loc[0, ["tp", "fp", "fn", "tn"]].tolist() == [1, 0, 0, 2]
    assert d.loc[1, ["tp", "fp", "fn", "tn"]].tolist() == [0, 1, 1, 1]
    assert d["k"].tolist() == [1, 1] and d["true_clicks"].tolist() == [1, 1]
    o = M.oracle_k_table(f).set_index("record_id")
    assert o.loc[0, "oracle_hits"] == 1 and o.loc[1, "oracle_hits"] == 0


def test_mean_with_ci_brackets_the_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    vals = rng.normal(0.6, 0.1, 300).tolist()
    persons = [f"u{i // 3}" for i in range(300)]
    e1 = M.mean_with_ci(vals, persons, seed=3)
    e2 = M.mean_with_ci(vals, persons, seed=3)
    assert e1 == e2 and e1.lo < e1.value < e1.hi
    assert e1.n_records == 300 and e1.n_persons == 100
    with_none = M.mean_with_ci([*vals, None], [*persons, "u0"], seed=3)
    assert with_none.n_records == 300


def test_clustered_persons_widen_the_interval():
    rng = np.random.default_rng(2)
    person_effect = rng.normal(0, 1, 20)
    vals = [person_effect[i // 10] for i in range(200)]
    clustered = M.mean_with_ci(vals, [f"u{i // 10}" for i in range(200)], seed=0)
    naive = M.mean_with_ci(vals, [f"u{i}" for i in range(200)], seed=0)
    assert (clustered.hi - clustered.lo) > 2 * (naive.hi - naive.lo)


def test_paired_difference_and_decision_cis():
    a = [0.7, 0.8, 0.9, None]
    b = [0.5, 0.6, 0.7, 0.4]
    d = M.paired_diff_with_ci(a, b, ["u1", "u1", "u2", "u3"], seed=0)
    assert d.value == pytest.approx(0.2) and d.n_records == 3
    table = M.decision_table(_frame())
    ci = M.decision_metrics_with_ci(table, ["u1", "u2"], seed=0)
    assert ci["precision"].value == pytest.approx(0.5)
    assert 0.0 <= ci["precision"].lo <= ci["precision"].hi <= 1.0
