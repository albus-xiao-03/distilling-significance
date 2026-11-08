"""E4 student: features, person split, training on label sources, recovery (synthetic)."""

import numpy as np
import pandas as pd
import pytest

from significance import baselines as B
from significance import episodes as E
from significance import student as S
from significance.data import records


def _setup(mind_frames):
    """Records for every synthetic impression of both files (about 35), as one set."""
    train, dev, news = mind_frames
    both = pd.concat([train, dev], ignore_index=True)
    recs = records.sample_records(both, len(both), 1, "evaluation")
    eps, dec = E.build_episodes(recs, train, dev)
    index = B.ExposureIndex([train, dev])
    return train, dev, news, recs, eps, dec.reset_index(drop=True), index


def test_numeric_features(mind_frames):
    _, _, news, recs, eps, dec, index = _setup(mind_frames)
    num = S.numeric_features(dec, eps, recs, news, index)
    assert list(num.columns) == list(S.NUMERIC) and len(num) == len(dec)
    assert num["past_category_share"].between(0, 1).all()
    ep = eps[0]
    rows = dec["record_id"] == ep.record_id
    cat = dict(zip(news["news_id"], news["category"], strict=True))
    past = [cat[n] for n in B.attended_before(ep)]
    first = dec[rows].iloc[0]
    expected = past.count(cat[first["news_id"]]) / max(len(past), 1)
    assert num[rows].iloc[0]["past_category_share"] == pytest.approx(expected)
    assert num[rows].iloc[0]["log_episode_size"] == pytest.approx(np.log1p(ep.n))


def test_split_by_person_keeps_persons_apart(mind_frames):
    *_, recs, _, _, _ = _setup(mind_frames)
    tr, te = S.split_by_person(recs, 0.3, seed=0)
    assert tr and te and not tr & te and tr | te == set(recs["record_id"])
    users = recs.set_index("record_id")["user_id"]
    assert not set(users.loc[list(tr)]) & set(users.loc[list(te)])


def test_distil_real_labels_beat_random_and_recovery(mind_frames):
    _, _, news, recs, eps, dec, index = _setup(mind_frames)
    # make the real decisions learnable: attended iff the category is "sports"
    cat = dict(zip(news["news_id"], news["category"], strict=True))
    dec = dec.assign(attended=dec["news_id"].map(lambda n: int(cat[n] == "sports")))
    num = S.numeric_features(dec, eps, recs, news, index)
    feats = S.Features(news, B.ContentModel(news))
    persons = recs.set_index("record_id")["user_id"]
    tr, te = S.split_by_person(recs, 0.3, seed=0)
    rng = np.random.default_rng(0)
    labels = {
        "real": dec["attended"].to_numpy(),
        "pseudo": dec["attended"].to_numpy(),
        "random": rng.integers(0, 2, len(dec)),
    }
    out = S.distil(dec, num, feats, persons, labels, tr, te, n_boot=30)
    assert out["students"]["real"]["auc"]["value"] > 0.9
    assert out["students"]["real"]["auc"]["value"] > out["students"]["random"]["auc"]["value"]
    assert out["recovery"]["value"] == pytest.approx(1.0)
    assert "recovery_random" in out


def test_one_class_labels_and_overlapping_splits_are_refused(mind_frames):
    _, _, news, recs, eps, dec, index = _setup(mind_frames)
    num = S.numeric_features(dec, eps, recs, news, index)
    feats = S.Features(news, B.ContentModel(news))
    persons = recs.set_index("record_id")["user_id"]
    ids = set(recs["record_id"])
    with pytest.raises(ValueError, match="both"):
        S.distil(dec, num, feats, persons, {"real": dec["attended"].to_numpy()}, ids, ids)
    tr, te = S.split_by_person(recs, 0.3, seed=0)
    with pytest.raises(ValueError, match="one class"):
        S.distil(dec, num, feats, persons, {"x": np.zeros(len(dec), dtype=int)}, tr, te)


def test_features_matrix_shape(mind_frames):
    _, _, news, recs, eps, dec, index = _setup(mind_frames)
    num = S.numeric_features(dec, eps, recs, news, index)
    content = B.ContentModel(news)
    feats = S.Features(news, content)
    x = feats.matrix(dec, num, fit_rows=np.ones(len(dec), dtype=bool))
    n_onehot = sum(len(c) for c in feats.encoder.categories_)
    assert x.shape == (len(dec), content.matrix.shape[1] + n_onehot + len(S.NUMERIC))
    assert isinstance(pd.Series(x.data).sum(), float)
