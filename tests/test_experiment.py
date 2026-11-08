"""Predictor assessment, paired comparison and strata (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from significance import episodes as E
from significance import experiment as X
from significance.data import records


def _setup(mind_frames):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=8, n_selection=6, n_probe=2, seed=3)
    eps, dec = E.build_episodes(sets["selection"], train, dev)
    persons = sets["selection"].set_index("record_id")["user_id"]
    return eps, dec, persons


def test_perfect_and_inverted_predictors(mind_frames):
    eps, dec, persons = _setup(mind_frames)
    good = X.predictor_frame(dec, dec["attended"] + 0.01, dec["attended"])
    res, per = X.assess(good, persons, n_boot=50)
    assert res["E1"]["auc"]["value"] == 1.0 and res["E1"]["ndcg10"]["value"] == 1.0
    # MIND's MRR averages 1/rank over all clicks: a perfect ranking of c clicks gives H(c)/c
    clicks = dec.groupby("record_id")["attended"].sum()
    perfect_mrr = np.mean([sum(1 / i for i in range(1, c + 1)) / c for c in clicks])
    assert res["E1"]["mrr"]["value"] == pytest.approx(perfect_mrr, abs=1e-4)
    assert res["E2"]["f1"]["value"] == 1.0 and res["E2"]["kappa"]["value"] == 1.0
    assert res["k"]["share_exact"] == 1.0 and res["oracle_k_precision"]["value"] == 1.0
    bad = X.predictor_frame(dec, 1 - dec["attended"], 1 - dec["attended"])
    res_bad, per_bad = X.assess(bad, persons, n_boot=50)
    assert res_bad["E1"]["auc"]["value"] == 0.0
    diff = X.compare(per, per_bad, persons, n_boot=50)
    assert diff["value"] == 1.0 and diff["records"] == len(eps)


def test_bgpt_frame_aligns_and_requires_coverage(mind_frames):
    _, dec, _ = _setup(mind_frames)
    preds = dec[["record_set", "record_id", "news_id"]].assign(
        confidence=np.linspace(0, 1, len(dec)), predicted_attention=0
    )
    frame = X.bgpt_frame(preds.sample(frac=1, random_state=0), dec)
    assert frame["news_id"].tolist() == dec["news_id"].tolist()
    assert frame["score"].tolist() == pytest.approx(preds["confidence"].tolist())
    with pytest.raises(ValueError, match="cover"):
        X.bgpt_frame(preds.iloc[:-1], dec)


def test_strata_and_by_stratum(mind_frames):
    eps, dec, persons = _setup(mind_frames)
    labels = X.strata(eps)
    assert set(labels) <= {"earlier impressions", "history clicks only", "no previous decisions"}
    cold = E.Episode("selection", 99, (), ("a",), 0, 0)
    hist = E.Episode("selection", 98, (E.EarlierEpisode(None, ("h",)),), ("a",), 0, 0)
    assert X.strata([cold, hist]).to_dict() == {
        99: "no previous decisions",
        98: "history clicks only",
    }
    _, per = X.assess(X.predictor_frame(dec, dec["attended"], dec["attended"]), persons, n_boot=20)
    out = X.by_stratum(per, labels, persons, n_boot=20)
    assert sum(v["records"] for v in out.values()) == len(eps)


def test_without_seen_before_drops_re_presented():
    frame = pd.DataFrame({"seen_before": [True, False, False], "x": [1, 2, 3]})
    assert X.without_seen_before(frame)["x"].tolist() == [2, 3]
