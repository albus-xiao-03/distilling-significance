"""BehaviorGPT as the teacher: ranking calls restricted to the presented observations.

One call (plans/R1.md 6.1):

    complete(H, limit=len(P), filters={"id": P}, catalog_id=<MIND catalog>, predict=...)

- `P` is split into chunks of at most 100 ids (the API's limit, verified in Phase 0); the
  chunks of one history share one score scale (Phase 0: differences <= 0.0003, against a
  score spread of about 0.04).
- No user_id is ever sent, so the SDK sends register_event=False and nothing is recorded
  for any person on the account.
- Every response is cached in SQLite under .temp/cache/ by (catalog, prediction question,
  history, chunk ids), so reruns and resumed runs make no new calls for what they have.
- 429 and 5xx answers and transport errors are retried with capped exponential backoff.
- A throttle caps call starts per second across worker threads.
- An id the API does not return (the filter silently drops unknown ids, Phase 0) is
  counted and left out of the scores; the decoder ranks it last.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from behaviorgpt import Order, View
from behaviorgpt._exceptions import UnboxAIError

from significance import config

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
CACHE_VERSION = 1


def to_sdk(sessions: Sequence[Sequence[tuple]]) -> list[list]:
    """Variant-writer sessions as SDK events (empty sessions dropped)."""
    out = []
    for session in sessions:
        events = []
        for kind, value in session:
            if kind == "view":
                events.append(View(value))
            elif kind == "order":
                events.append(Order(list(value)))
            else:
                raise ValueError(f"unknown event kind {kind!r}")
        if events:
            out.append(events)
    return out


def history_key(sessions: Sequence[Sequence[tuple]]) -> list:
    """A canonical JSON-able form of a history (empty sessions dropped)."""
    return [
        [[kind, list(value) if kind == "order" else value] for kind, value in s]
        for s in sessions
        if s
    ]


class ResponseCache:
    """SQLite cache of ranking responses: key -> {news_id: score}. Thread-safe."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, scores TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, float] | None:
        with self._lock:
            row = self._db.execute("SELECT scores FROM responses WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, scores: dict[str, float]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                (key, json.dumps(scores), time.time()),
            )
            self._db.commit()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone()
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Throttle:
    """At most `rate` call starts per second, shared by all threads (None: no limit)."""

    def __init__(
        self,
        rate: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate is not None and rate <= 0:
            raise ValueError("rate must be positive")
        self.interval = 0.0 if rate is None else 1.0 / rate
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            self._sleep(start - now)


@dataclass
class CallStats:
    """What the teacher did: API calls made, cache hits, retries, ids not returned."""

    api_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    missing_ids: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + value)

    def as_dict(self) -> dict[str, int]:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "missing_ids": self.missing_ids,
        }


class Teacher:
    """Scores presented observations after a history, through BehaviorGPT, unchanged."""

    def __init__(
        self,
        client,
        catalog_id: str,
        *,
        cache: ResponseCache | None = None,
        throttle: Throttle | None = None,
        retries: int = 6,
        max_ids: int = config.MAX_IDS_PER_CALL,
        backoff_cap: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        offline: bool = False,
    ):
        if not 1 <= max_ids <= config.MAX_IDS_PER_CALL:
            raise ValueError(f"max_ids must be between 1 and {config.MAX_IDS_PER_CALL}")
        self.client = client
        self.catalog_id = catalog_id
        self.cache = cache
        self.throttle = throttle or Throttle(None)
        self.retries = retries
        self.max_ids = max_ids
        self.backoff_cap = backoff_cap
        self._sleep = sleep
        self.offline = offline  # cache only: a miss raises instead of calling the API
        self.stats = CallStats()

    def chunks(self, candidates: Sequence[str]) -> list[list[str]]:
        ids = list(candidates)
        return [ids[i : i + self.max_ids] for i in range(0, len(ids), self.max_ids)]

    def cache_key(self, sessions: Sequence[Sequence[tuple]], chunk: Sequence[str], predict: str):
        raw = json.dumps(
            {
                "v": CACHE_VERSION,
                "catalog": self.catalog_id,
                "predict": predict,
                "history": history_key(sessions),
                "ids": sorted(chunk),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def score(
        self, sessions: Sequence[Sequence[tuple]], candidates: Sequence[str], predict: str = "view"
    ) -> dict[str, float]:
        """BehaviorGPT's scores for `candidates` after `sessions` (one shared scale). Ids
        the API does not return are absent from the result."""
        if predict not in ("view", "purchase"):
            raise ValueError(f"predict must be 'view' or 'purchase', not {predict!r}")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates must be distinct")
        scores: dict[str, float] = {}
        for chunk in self.chunks(candidates):
            scores.update(self._score_chunk(sessions, chunk, predict))
        missing = len(candidates) - len(scores)
        if missing:
            self.stats.add(missing_ids=missing)
        return scores

    def _score_chunk(self, sessions, chunk: list[str], predict: str) -> dict[str, float]:
        key = self.cache_key(sessions, chunk, predict)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.add(cache_hits=1)
                return hit
        if self.offline:
            raise LookupError("offline teacher: response not in the cache")
        events = to_sdk(sessions)
        wanted = set(chunk)
        for attempt in range(self.retries + 1):
            self.throttle.wait()
            try:
                res = self.client.complete(
                    events,
                    limit=len(chunk),
                    catalog_id=self.catalog_id,
                    filters={"id": list(chunk)},
                    predict=predict,
                )
            except UnboxAIError as exc:
                if exc.status_code not in RETRYABLE_STATUS or attempt == self.retries:
                    raise
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            else:
                self.stats.add(api_calls=1)
                result = {it.id: float(it.score) for it in res.items if it.id in wanted}
                if self.cache is not None:
                    self.cache.put(key, result)
                return result
            self.stats.add(retries=1)
            self._sleep(min(2.0**attempt, self.backoff_cap) * (0.5 + random.random() / 2))
        raise AssertionError("unreachable")
