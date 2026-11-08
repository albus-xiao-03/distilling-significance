"""Next-action decoder with a fake teacher (plans/R1.md 6.2-6.4)."""

import pytest

from fakes import FakeClient, history_items
from significance import episodes as E
from significance.teacher import api, decode

EP = E.Episode(
    record_set="selection",
    record_id=7,
    previous=(E.EarlierEpisode(None, ("h1",)), E.EarlierEpisode(("p1", "p2"), ("p2",))),
    presented=("a", "b", "c", "d", "e"),
    earlier_presented=2,
    earlier_attended=1,  # rate 0.5 -> k = round(2.5) = 3
)

BASE = {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.4, "e": 0.2}


def boost_after_chosen(history, item, predict):
    """b is best first; once b is in the history, e jumps to the top; then base order."""
    seen = {x for _, x in history_items(history)}
    if "b" in seen and item == "e" and "e" not in seen:
        return 0.9
    return BASE[item]


def _teacher(score_fn=boost_after_chosen, **kw):
    client = FakeClient(score_fn=score_fn, **kw)
    return api.Teacher(client, "cat", sleep=lambda s: None), client


def test_v1_decodes_k_steps_and_rescores_after_each_choice():
    teacher, client = _teacher()
    out = decode.decode(teacher, EP, "V1")
    assert out.k == 3
    # step 1: b (0.5); step 2: e jumps to 0.9; step 3: d (0.4); rest by last score: c, a
    assert out.ranking == ("b", "e", "d", "c", "a")
    assert len(client.calls) == 3
    assert [len(c["ids"]) for c in client.calls] == [5, 4, 3]
    assert out.requests == 3
    rows = out.rows()
    assert [r["predicted_attention"] for r in rows] == [1, 1, 1, 0, 0]
    assert [r["attention_rank"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["confidence"] for r in rows] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2])
    assert {r["variant"] for r in rows} == {"V1"}


def test_current_session_grows_in_the_variant_form():
    teacher, client = _teacher()
    decode.decode(teacher, EP, "V2")
    last_sessions = [s for s in client.calls[2]["history"]]
    current = last_sessions[-1]
    assert [type(e).__name__ for e in current] == ["View", "View", "Order"]
    assert [c.item for c in current[-1].items] == ["b", "e"]
    teacher, client = _teacher()
    decode.decode(teacher, EP, "V1")
    assert [type(e).__name__ for e in client.calls[2]["history"][-1]] == ["View", "View"]


def test_current_session_never_holds_an_unchosen_candidate():
    """Even when earlier sessions hold re-presented candidates (L7, by design), the current
    session only ever holds the observations BehaviorGPT already chose."""
    ep = E.Episode(
        "selection",
        8,
        (E.EarlierEpisode(("a", "p1"), ("p1",)),),  # a was shown before and ignored
        ("a", "b", "c", "d", "e"),
        2,
        1,
    )
    teacher, client = _teacher()
    out = decode.decode(teacher, ep, "V2")
    for step, call in enumerate(client.calls):
        current = {x for _, x in history_items([call["history"][-1]])} if step else set()
        assert current == set(out.ranking[:step])
        assert not current & set(call["ids"])
    first_history = {x for _, x in history_items(client.calls[0]["history"])}
    assert "a" in first_history  # the earlier encounter is kept


def test_empty_history_sends_only_the_current_session():
    teacher, client = _teacher()
    out = decode.decode(teacher, EP, "V1", empty_history=True)
    assert client.calls[0]["history"] == []
    assert out.history_events == 0 and out.empty_history


def test_k_one_is_a_single_ranking_and_k_capped_at_n():
    teacher, client = _teacher()
    out = decode.decode(teacher, EP, "V1o", k=1)
    assert out.ranking == ("b", "d", "c", "e", "a") and len(client.calls) == 1
    teacher, _ = _teacher()
    assert decode.decode(teacher, EP, "V1", k=99).k == 5


def test_v3_one_purchase_ranking_of_the_screen():
    teacher, client = _teacher()
    out = decode.decode(teacher, EP, "V3", k=2)
    assert len(client.calls) == 1 and client.calls[0]["predict"] == "purchase"
    screen = client.calls[0]["history"][-1]
    assert sorted(e.item for e in screen) == sorted(EP.presented)
    assert out.ranking == ("b", "d", "c", "e", "a")


def test_missing_ids_rank_last_and_ties_break_by_hash():
    teacher, _ = _teacher(drop={"b"})
    out = decode.decode(teacher, EP, "V1", k=1)
    assert out.ranking[-1] == "b" and out.missing == ("b",)
    assert out.rows()[-1]["score"] is None
    flat, _ = _teacher(score_fn=lambda h, i, p: 0.5)
    r1 = decode.decode(flat, EP, "V1", k=1).ranking
    assert sorted(r1) == sorted(EP.presented)
    assert r1 == decode.decode(_teacher(score_fn=lambda h, i, p: 0.5)[0], EP, "V1", k=1).ranking


def test_requests_needed_counts_chunks_per_step():
    assert decode.requests_needed(5, 3, "V1") == 3
    assert decode.requests_needed(150, 2, "V2") == 4  # 150 -> 2 chunks, 149 -> 2 chunks
    assert decode.requests_needed(101, 2, "V1") == 3  # 101 -> 2, 100 -> 1
    assert decode.requests_needed(250, 1, "V3") == 3
    assert decode.requests_needed(3, 9, "V1") == 3  # k capped at n
    with pytest.raises(ValueError):
        decode.requests_needed(3, 1, "V7")


def test_decoder_matches_requests_needed_on_large_episodes():
    big = E.Episode("selection", 1, (), tuple(f"N{i}" for i in range(230)), 100, 2)
    teacher, client = _teacher(score_fn=lambda h, i, p: hash(i) % 1000 / 1000)
    out = decode.decode(teacher, big, "V1")
    assert out.k == 5  # round(0.02 * 230) = round(4.6) = 5
    assert len(client.calls) == decode.requests_needed(230, 5, "V1") == out.requests
