"""Decision episodes: what BehaviorGPT may see, and the real decisions kept apart.

A record is one MIND impression: the observations presented to a person (headlines), which
of them the person chose to open (the real attention decisions), and the person's previous
decisions (plans/R1.md 5.1, 5.3):

- the History clicks: MIND's click history from the first four weeks, oldest first, with
  no presented sets; constant per person and identical across both files (Phase 1);
- the person's earlier impressions in the TRAINING file on days strictly before the
  record's day, oldest first, each with its presented set and its clicks.

The day-level cutoff makes the selection records (training file, last day) and the
evaluation records (validation file, the following day) see their history the same way,
and it keeps every record's own day out of every history: no record's real decisions ever
reach BehaviorGPT through another record's history (plans/R1.md 7.1). Validation-file
impressions are never used as previous decisions.

The real decisions are stored in a separate evaluation table, never in an Episode.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from significance import config


@dataclass(frozen=True)
class EarlierEpisode:
    """One earlier decision episode: `presented` is None for the History clicks."""

    presented: tuple[str, ...] | None
    attended: tuple[str, ...]


@dataclass(frozen=True)
class Episode:
    """What a record lets BehaviorGPT see. The real decisions are not part of it."""

    record_set: str
    record_id: int
    previous: tuple[EarlierEpisode, ...]  # oldest first; History clicks first
    presented: tuple[str, ...]  # the new observations to decide on
    earlier_presented: int  # observations presented in earlier impressions
    earlier_attended: int  # of which the person attended

    @property
    def n(self) -> int:
        return len(self.presented)

    @property
    def has_earlier_impressions(self) -> bool:
        return any(e.presented is not None for e in self.previous)

    @property
    def history_clicks(self) -> int:
        return sum(len(e.attended) for e in self.previous if e.presented is None)


def _attended(presented: Sequence[str], clicked: Sequence[int]) -> tuple[str, ...]:
    """Clicked ids in presented order (MIND records no order between the clicks of one
    impression; its presented order is already shuffled)."""
    return tuple(n for n, c in zip(presented, clicked, strict=True) if c)


def build_episodes(
    records: pd.DataFrame, train: pd.DataFrame, dev: pd.DataFrame
) -> tuple[list[Episode], pd.DataFrame]:
    """Episodes for `records`, and their real decisions as a separate table.

    The decisions table has one row per presented observation: record_set, record_id,
    news_id, attended (the real decision) and, from previous decisions only, seen_before
    (presented or clicked in an earlier episode) and attended_before.
    """
    by_split = {"train": train.set_index("impression_id"), "dev": dev.set_index("impression_id")}
    earlier_by_user = {
        user: frame.sort_values(["time", "impression_id"])
        for user, frame in train.groupby("user_id", sort=False)
    }
    empty = train.iloc[0:0]
    episodes: list[Episode] = []
    decision_rows: list[tuple] = []
    for rec in records.itertuples(index=False):
        row = by_split[rec.split].loc[rec.impression_id]
        if row["user_id"] != rec.user_id:
            raise ValueError(f"record {rec.record_id}: user does not match the impression")
        previous: list[EarlierEpisode] = []
        if row["history"]:
            previous.append(EarlierEpisode(None, tuple(row["history"])))
        earlier = earlier_by_user.get(rec.user_id, empty)
        earlier = earlier[earlier["date"] < row["date"]]
        n_pres = n_att = 0
        for e in earlier.itertuples(index=False):
            previous.append(EarlierEpisode(tuple(e.presented), _attended(e.presented, e.clicked)))
            n_pres += len(e.presented)
            n_att += sum(e.clicked)
        ep = Episode(
            record_set=rec.record_set,
            record_id=int(rec.record_id),
            previous=tuple(previous),
            presented=tuple(row["presented"]),
            earlier_presented=n_pres,
            earlier_attended=n_att,
        )
        episodes.append(ep)
        seen = {n for p in previous for n in (p.presented or ())} | {
            n for p in previous for n in p.attended
        }
        attended_before = {n for p in previous for n in p.attended}
        for news_id, clicked in zip(row["presented"], row["clicked"], strict=True):
            decision_rows.append(
                (
                    rec.record_set,
                    int(rec.record_id),
                    news_id,
                    int(clicked),
                    news_id in seen,
                    news_id in attended_before,
                )
            )
    decisions = pd.DataFrame(
        decision_rows,
        columns=[
            "record_set",
            "record_id",
            "news_id",
            "attended",
            "seen_before",
            "attended_before",
        ],
    )
    return episodes, decisions


# --- storage under .temp/work/episodes/ ----------------------------------------------


def episodes_dir(work: Path) -> Path:
    return work / "episodes"


def save_episodes(episodes: Iterable[Episode], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep)) + "\n")
    return path


def load_episodes(path: Path) -> list[Episode]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            d["previous"] = tuple(
                EarlierEpisode(
                    tuple(p["presented"]) if p["presented"] is not None else None,
                    tuple(p["attended"]),
                )
                for p in d["previous"]
            )
            d["presented"] = tuple(d["presented"])
            out.append(Episode(**d))
    return out


# --- representation variants (plans/R1.md 5.4) --------------------------------------
#
# Events are written as plain tuples, ("view", news_id) or ("order", (news_id, ...)); the
# teacher turns them into SDK View/Order events. An attention decision "attend" is the
# person's action on the observation; "not attend" is the documented way of expressing a
# non-choice: leaving the observation out (V1, V1o) or leaving its View out of the Order
# (V2, V3) (plans/R1.md L4).
#
#   V1   earlier episode -> View each attended observation
#   V1o  earlier episode -> View each attended observation, then Order them
#   V2   earlier episode -> View every presented observation, then Order the attended ones
#   V3   V2's history plus the current screen as Views; predict="purchase" (Phase 0: no-go)
#
# The History clicks have no presented set: they are Views (V1), or Views then one Order of
# them (V1o, V2, V3), like an episode in which every observation was attended.
#
# During next-action decoding the observations BehaviorGPT has chosen so far form the
# current session in the same form: V1 views them; V1o and V2 view them and then order them.
# The current session never holds a candidate before BehaviorGPT chooses it (V3's screen
# aside). Earlier sessions are written as they happened, so a candidate the person attended
# earlier (V1, V1o) or was shown earlier (V2, V3) stays in them: plans/R1.md L7 expects such
# re-presented observations in V2's history and reports them separately (the seen_before
# flag of the decisions table; 0.6 % of evaluation and 4.7 % of selection observations).

Event = tuple[str, "str | tuple[str, ...]"]
Session = list[Event]
ORDER_VARIANTS = frozenset({"V1o", "V2", "V3"})
ALL_VARIANTS = ("V1", "V1o", "V2", "V3")


def check_variant(variant: str) -> str:
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
    return variant


def _observations(ep: EarlierEpisode, variant: str) -> list[tuple[str, bool]]:
    """An earlier episode as (observation, attended) pairs in the variant's form."""
    if variant in ("V1", "V1o") or ep.presented is None:
        return [(a, True) for a in ep.attended]
    attended = set(ep.attended)
    return [(p, p in attended) for p in ep.presented]


def _render(obs: Sequence[tuple[str, bool]], with_order: bool) -> Session:
    events: Session = [("view", x) for x, _ in obs]
    attended = tuple(x for x, a in obs if a)
    if with_order and attended:
        events.append(("order", attended))
    return events


def _fit(obs: Sequence[tuple[str, bool]], with_order: bool, budget: int) -> Session:
    """The session cut to at most `budget` events: the oldest not-attended observations
    are dropped first, then the oldest attended ones; an Order lists the attended
    observations that stay."""
    has_order = with_order and any(a for _, a in obs)
    room = budget - (1 if has_order else 0)
    if room <= 0:
        return []
    keep = list(obs)
    while len(keep) > room:
        keep.pop(next((j for j, (_, a) in enumerate(keep) if not a), 0))
    return _render(keep, with_order)


def previous_sessions(episode: Episode, variant: str, budget: int) -> list[Session]:
    """The episode's previous decisions in `variant`'s form, oldest first, in at most
    `budget` events: the most recent sessions are kept whole, the first one that does not
    fit is cut (`_fit`), and older ones are dropped (plans/R1.md 5.4, limit L10)."""
    check_variant(variant)
    with_order = variant in ORDER_VARIANTS
    out: list[Session] = []
    remaining = budget
    for ep in reversed(episode.previous):
        obs = _observations(ep, variant)
        if not obs:
            continue
        full = _render(obs, with_order)
        if len(full) <= remaining:
            out.append(full)
            remaining -= len(full)
            continue
        cut = _fit(obs, with_order, remaining)
        if cut:
            out.append(cut)
        break
    out.reverse()
    return out


def current_session(variant: str, chosen: Sequence[str]) -> Session:
    """The current session once BehaviorGPT has chosen `chosen` (V1, V1o, V2)."""
    check_variant(variant)
    if variant == "V3":
        raise ValueError("V3's current session is the screen: use screen_session")
    events: Session = [("view", x) for x in chosen]
    if variant in ORDER_VARIANTS and chosen:
        events.append(("order", tuple(chosen)))
    return events


def screen_session(episode: Episode) -> Session:
    """V3: the current presented observations as Views, in an order seeded by the record
    (MIND's own presented order is shuffled and carries no position)."""
    order = list(episode.presented)
    random.Random(f"{episode.record_set}:{episode.record_id}").shuffle(order)
    return [("view", x) for x in order]


def current_budget(variant: str, k: int, n: int) -> int:
    """The most events the current session holds: at the last of k decoding steps for V1,
    V1o and V2 (k - 1 chosen, plus their Order); the whole screen for V3."""
    check_variant(variant)
    if variant == "V3":
        return n
    chosen = max(k - 1, 0)
    return chosen + (1 if variant in ORDER_VARIANTS and chosen else 0)


def history_for(
    episode: Episode,
    variant: str,
    k: int,
    *,
    max_events: int = config.MAX_HISTORY_EVENTS,
    empty_history: bool = False,
) -> list[Session]:
    """The previous sessions sent for this record: capped so that they and the largest
    current session fit in `max_events` (K). `empty_history` drops them (E3a)."""
    if empty_history:
        return []
    budget = max_events - current_budget(variant, k, episode.n)
    return previous_sessions(episode, variant, max(budget, 0))


def count_events(sessions: Iterable[Session]) -> int:
    return sum(len(s) for s in sessions)


# --- how many observations get attention: k (plans/R1.md 6.3, limit L3, decision N5) --


def attention_k(n_presented: int, earlier_presented: int, earlier_attended: int) -> int:
    """k = max(1, round(a * n)), with a the person's own attention rate in their earlier
    episodes (attended / presented) and n the observations presented now; k = 1 without
    earlier episodes. Rounds half up (not Python's half-to-even) and never exceeds n.
    Uses previous decisions only, never the episode being predicted."""
    if n_presented < 1:
        raise ValueError("an episode presents at least one observation")
    if earlier_presented < 0 or not 0 <= earlier_attended <= earlier_presented:
        raise ValueError("earlier counts must satisfy 0 <= attended <= presented")
    if earlier_presented == 0:
        return 1
    # round(a * n) half up, in integers: floats turn e.g. 3/94 * 47 = 1.5 into 1.4999...
    half_up = (2 * earlier_attended * n_presented + earlier_presented) // (2 * earlier_presented)
    return min(n_presented, max(1, half_up))


def episode_k(episode: Episode) -> int:
    return attention_k(episode.n, episode.earlier_presented, episode.earlier_attended)
