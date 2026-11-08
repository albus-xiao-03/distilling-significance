"""The k rule (plans/R1.md 6.3; N5)."""

import pytest

from significance import episodes as E


@pytest.mark.parametrize(
    ("n", "pres", "att", "k"),
    [
        (10, 0, 0, 1),  # no earlier episodes -> 1
        (40, 100, 4, 2),  # 0.04 * 40 = 1.6 -> 2
        (25, 100, 4, 1),  # 0.04 * 25 = 1.0 -> 1
        (10, 100, 1, 1),  # 0.1 rounds to 0 -> minimum 1
        (50, 100, 5, 3),  # 2.5 rounds half up -> 3 (Python's round would give 2)
        (30, 100, 5, 2),  # 1.5 -> 2
        (4, 10, 10, 4),  # rate 1 -> all, never more than n
        (3, 2, 2, 3),
        (47, 94, 3, 2),  # 3/94 * 47 = 1.5 exactly; floating point would give 1
        (45, 10, 7, 32),  # 31.5 exactly -> 32
    ],
)
def test_attention_k(n, pres, att, k):
    assert E.attention_k(n, pres, att) == k


def test_attention_k_validates_inputs():
    with pytest.raises(ValueError):
        E.attention_k(0, 10, 1)
    with pytest.raises(ValueError):
        E.attention_k(5, 10, 11)
    with pytest.raises(ValueError):
        E.attention_k(5, -1, 0)


def test_episode_k_uses_previous_decisions_only():
    ep = E.Episode("evaluation", 0, (), tuple(f"c{i}" for i in range(40)), 100, 4)
    assert E.episode_k(ep) == 2
    no_earlier = E.Episode("evaluation", 1, (E.EarlierEpisode(None, ("h",)),), ("a", "b"), 0, 0)
    assert E.episode_k(no_earlier) == 1  # History clicks carry no presented set
