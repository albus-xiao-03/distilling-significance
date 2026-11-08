"""Variant writers V1, V1o, V2, V3 and the history cap K (hand-built and synthetic episodes)."""

import pytest

from significance import episodes as E
from significance.data import records

EP = E.Episode(
    record_set="evaluation",
    record_id=3,
    previous=(
        E.EarlierEpisode(None, ("h1", "h2")),
        E.EarlierEpisode(("p1", "p2", "p3"), ("p2",)),
        E.EarlierEpisode(("q1", "q2"), ("q1", "q2")),
    ),
    presented=("c1", "c2", "c3", "c4"),
    earlier_presented=5,
    earlier_attended=3,
)


def v(x):
    return ("view", x)


def o(*xs):
    return ("order", tuple(xs))


def test_v1_views_attended_observations_one_session_per_episode():
    assert E.previous_sessions(EP, "V1", 100) == [
        [v("h1"), v("h2")],
        [v("p2")],
        [v("q1"), v("q2")],
    ]


def test_v1o_adds_an_order_of_the_attended():
    assert E.previous_sessions(EP, "V1o", 100) == [
        [v("h1"), v("h2"), o("h1", "h2")],
        [v("p2"), o("p2")],
        [v("q1"), v("q2"), o("q1", "q2")],
    ]


def test_v2_views_everything_presented_and_orders_the_attended():
    expected = [
        [v("h1"), v("h2"), o("h1", "h2")],
        [v("p1"), v("p2"), v("p3"), o("p2")],
        [v("q1"), v("q2"), o("q1", "q2")],
    ]
    assert E.previous_sessions(EP, "V2", 100) == expected
    assert E.previous_sessions(EP, "V3", 100) == expected


def test_current_session_forms():
    assert E.current_session("V1", []) == []
    assert E.current_session("V1", ["c2", "c4"]) == [v("c2"), v("c4")]
    assert E.current_session("V1o", ["c2"]) == [v("c2"), o("c2")]
    assert E.current_session("V2", ["c2", "c4"]) == [v("c2"), v("c4"), o("c2", "c4")]
    with pytest.raises(ValueError, match="screen"):
        E.current_session("V3", ["c1"])
    with pytest.raises(ValueError, match="unknown variant"):
        E.current_session("V9", [])


def test_screen_session_is_a_seeded_permutation():
    s1, s2 = E.screen_session(EP), E.screen_session(EP)
    assert s1 == s2
    assert sorted(x for _, x in s1) == sorted(EP.presented)


def _items(sessions):
    return {x for s in sessions for e in s for x in ((e[1],) if e[0] == "view" else e[1])}


def test_written_history_holds_no_candidate_without_an_earlier_encounter():
    for variant in ("V1", "V1o", "V2"):
        assert not _items(E.previous_sessions(EP, variant, 100)) & set(EP.presented)


def test_re_presented_candidates_stay_in_earlier_sessions_by_design():
    """L7: earlier sessions are written as they happened. c1 was clicked before (History),
    c2 was shown before and ignored; both are presented again now."""
    ep = E.Episode(
        "selection",
        5,
        (E.EarlierEpisode(None, ("c1",)), E.EarlierEpisode(("c2", "p1", "p2"), ("p1",))),
        ("c1", "c2", "c3"),
        3,
        1,
    )
    assert _items(E.previous_sessions(ep, "V1", 100)) & set(ep.presented) == {"c1"}
    assert _items(E.previous_sessions(ep, "V1o", 100)) & set(ep.presented) == {"c1"}
    assert _items(E.previous_sessions(ep, "V2", 100)) & set(ep.presented) == {"c1", "c2"}


def test_cap_keeps_recent_sessions_whole_and_cuts_the_next():
    # V2 sessions are 3 (History), 4 (p) and 3 (q) events
    assert E.previous_sessions(EP, "V2", 7) == [
        [v("p1"), v("p2"), v("p3"), o("p2")],
        [v("q1"), v("q2"), o("q1", "q2")],
    ]
    # budget 6: q whole (3), p cut to 3 events: oldest not-attended (p1) dropped first
    assert E.previous_sessions(EP, "V2", 6) == [
        [v("p2"), v("p3"), o("p2")],
        [v("q1"), v("q2"), o("q1", "q2")],
    ]
    # budget 4: q whole, p needs Order + 1 view at least -> [p2, Order]
    assert E.previous_sessions(EP, "V2", 4) == [[v("q1"), v("q2"), o("q1", "q2")]]
    assert E.previous_sessions(EP, "V2", 5) == [
        [v("p2"), o("p2")],
        [v("q1"), v("q2"), o("q1", "q2")],
    ]
    assert E.previous_sessions(EP, "V2", 0) == []


def test_history_clicks_are_cut_to_the_most_recent():
    ep = E.Episode(
        "selection", 0, (E.EarlierEpisode(None, ("a", "b", "c", "d", "e")),), ("x",), 0, 0
    )
    assert E.previous_sessions(ep, "V1", 3) == [[v("c"), v("d"), v("e")]]
    assert E.previous_sessions(ep, "V1o", 3) == [[v("d"), v("e"), o("d", "e")]]


def test_current_budget_and_history_for():
    assert E.current_budget("V1", 1, 4) == 0
    assert E.current_budget("V1", 3, 4) == 2
    assert E.current_budget("V2", 3, 4) == 3
    assert E.current_budget("V1o", 1, 4) == 0
    assert E.current_budget("V3", 1, 4) == 4
    assert E.history_for(EP, "V2", 1, max_events=7) == E.previous_sessions(EP, "V2", 7)
    assert E.history_for(EP, "V2", 3, max_events=10) == E.previous_sessions(EP, "V2", 7)
    assert E.history_for(EP, "V2", 1, empty_history=True) == []


def test_history_plus_current_session_never_exceeds_k(mind_frames):
    train, dev, _ = mind_frames
    sets = records.build_record_sets(train, dev, n_evaluation=8, n_selection=6, n_probe=2, seed=4)
    eps, _ = E.build_episodes(sets["selection"], train, dev)
    for ep in eps:
        for variant in ("V1", "V1o", "V2"):
            for k in (1, 2, 4):
                for cap in (5, 12, 40):
                    hist = E.history_for(ep, variant, k, max_events=cap)
                    chosen = list(ep.presented[: k - 1])
                    total = E.count_events(hist) + len(E.current_session(variant, chosen))
                    assert total <= cap
