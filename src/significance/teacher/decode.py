"""Next-action decoding: BehaviorGPT acts as the person (plans/R1.md 6.2-6.4).

V1, V1o, V2:
  1. score the presented observations after the person's history; the top one is
     BehaviorGPT's predicted next action, so it is attended;
  2. add that attention event to the current session (V1: View; V1o, V2: View, then an
     Order of everything chosen so far) and score the rest again;
  3. repeat until k observations are attended (the k rule, limit L3). The rest are not
     attended and keep the order of their last scores.
V3 (Phase 0 no-go, off by default): one predict="purchase" ranking of the screen; the top
  k are attended.

Per observation: predicted_attention (1/0), attention_rank (1 = BehaviorGPT's first
predicted action) and confidence = 1 - (attention_rank - 1) / n, rank-based because the
API's scores are ranking scores, not probabilities (L2). Exact ties are broken by a hash
of the record and the item, never by MIND's presented order.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from significance import config
from significance import episodes as E

NEG_INF = float("-inf")


def tiebreak(record_set: str, record_id: int, item: str) -> float:
    digest = hashlib.sha256(f"{record_set}:{record_id}:{item}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


@dataclass(frozen=True)
class Decoded:
    record_set: str
    record_id: int
    variant: str
    empty_history: bool
    k: int
    ranking: tuple[str, ...]  # attention order: chosen ones first, then by last score
    scores: dict[str, float | None]  # each observation's last score (None: not returned)
    requests: int  # ranking requests (chunks) the record needed, cached or not
    history_events: int  # events of the previous sessions sent

    @property
    def n(self) -> int:
        return len(self.ranking)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(x for x in self.ranking if self.scores[x] is None)

    def rows(self) -> list[dict]:
        out = []
        for rank, news_id in enumerate(self.ranking, 1):
            out.append(
                {
                    "record_set": self.record_set,
                    "record_id": self.record_id,
                    "news_id": news_id,
                    "attention_rank": rank,
                    "predicted_attention": int(rank <= self.k),
                    "confidence": 1.0 - (rank - 1) / self.n,
                    "score": self.scores[news_id],
                    "k": self.k,
                    "variant": self.variant,
                    "empty_history": self.empty_history,
                }
            )
        return out


def requests_needed(n: int, k: int, variant: str, max_ids: int = config.MAX_IDS_PER_CALL) -> int:
    """Ranking requests one record needs: V1/V1o/V2 score n, n-1, ..., n-k+1 observations
    in k steps, each in chunks of at most `max_ids`; V3 scores n once."""
    E.check_variant(variant)
    k = min(k, n)
    if variant == "V3":
        return math.ceil(n / max_ids)
    return sum(math.ceil((n - j) / max_ids) for j in range(k))


def decode(
    teacher,
    episode: E.Episode,
    variant: str,
    *,
    k: int | None = None,
    empty_history: bool = False,
    max_events: int = config.MAX_HISTORY_EVENTS,
) -> Decoded:
    """BehaviorGPT's predicted attention decisions for one episode."""
    E.check_variant(variant)
    n = episode.n
    k = min(E.episode_k(episode) if k is None else k, n)
    if k < 1:
        raise ValueError("k must be at least 1")
    history = E.history_for(episode, variant, k, max_events=max_events, empty_history=empty_history)
    key = (episode.record_set, episode.record_id)

    def order(items: Sequence[str], last: dict[str, float]) -> list[str]:
        return sorted(items, key=lambda x: (-last[x], tiebreak(*key, x)))

    last: dict[str, float] = {}
    requests = 0
    if variant == "V3":
        scores = teacher.score([*history, E.screen_session(episode)], episode.presented, "purchase")
        requests += len(teacher.chunks(episode.presented))
        last = {x: scores.get(x, NEG_INF) for x in episode.presented}
        ranking = order(episode.presented, last)
    else:
        remaining = list(episode.presented)
        chosen: list[str] = []
        for _ in range(k):
            current = E.current_session(variant, chosen)
            sessions = [*history, current] if current else list(history)
            scores = teacher.score(sessions, remaining, "view")
            requests += len(teacher.chunks(remaining))
            for x in remaining:
                last[x] = scores.get(x, NEG_INF)
            best = order(remaining, last)[0]
            chosen.append(best)
            remaining.remove(best)
        ranking = chosen + order(remaining, last)
    return Decoded(
        record_set=episode.record_set,
        record_id=episode.record_id,
        variant=variant,
        empty_history=empty_history,
        k=k,
        ranking=tuple(ranking),
        scores={x: (None if last[x] == NEG_INF else last[x]) for x in ranking},
        requests=requests,
        history_events=E.count_events(history),
    )
