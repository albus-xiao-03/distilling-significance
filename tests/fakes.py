"""A fake BehaviorGPT client: deterministic scores, no network (tests only)."""

from __future__ import annotations

import hashlib
import json

from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.types import Order, UnboxAIResponse, View
from behaviorgpt.types.http import Item


def history_items(history) -> list[tuple[str, str]]:
    """(kind, item) for every event of an SDK history (list of sessions)."""
    out = []
    for session in history:
        for ev in session:
            if isinstance(ev, View):
                out.append(("view", ev.item))
            elif isinstance(ev, Order):
                out.extend(("order", ci.item) for ci in ev.items)
    return out


def hashed_score(history, item: str, predict: str) -> float:
    raw = json.dumps([history_items(history), item, predict])
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) / 2**32


class FakeClient:
    """Stands in for UnboxAIClient.complete with the id filter.

    `score_fn(history, item, predict)` gives each score (default: a hash of all three, so
    every change to the history changes the scores). `fail` is a list of exceptions raised
    by the first calls, in order. `drop` ids are never returned, like unknown ids.
    """

    def __init__(self, score_fn=hashed_score, fail=(), drop=()):
        self.score_fn = score_fn
        self.fail = list(fail)
        self.drop = set(drop)
        self.calls = []

    def complete(
        self,
        history=(),
        limit=10,
        offset=0,
        *,
        catalog_id=None,
        filters=None,
        predict="view",
        user_id=None,
        register_event=None,
        **kwargs,
    ):
        assert user_id is None and not register_event, "the teacher never sends a user"
        ids = list(filters["id"])
        assert 1 <= len(ids) <= 100 and limit == len(ids) and offset == 0
        self.calls.append(
            {"history": history, "ids": ids, "predict": predict, "catalog_id": catalog_id}
        )
        if self.fail:
            raise self.fail.pop(0)
        items = [
            Item(id=i, data={}, score=self.score_fn(history, i, predict))
            for i in ids
            if i not in self.drop
        ]
        items.sort(key=lambda it: -it.score)
        return UnboxAIResponse(items=items[:limit], offset=0, limit=limit)


def http_error(status: int) -> UnboxAIError:
    return UnboxAIError(f"status {status}", status_code=status)
