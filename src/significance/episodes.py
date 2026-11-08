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
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


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
