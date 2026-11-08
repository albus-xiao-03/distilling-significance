"""Prediction runs: estimate, run and choose the variant (plans/R1.md 6.5, 9 Phase 4).

- `estimate` counts the exact ranking requests a run needs (k steps x chunks per record,
  plans/R1.md 6.1-6.3) and how many of them the cache already holds where that is known
  in advance (the first decoding step).
- `run_variant` decodes every record of a set with one variant, in parallel threads; the
  response cache makes a rerun resume where a stopped run left off.
- `choose_variant` picks the variant with the best mean per-record AUC on the selection
  records (D16); ties go to the simpler variant.

Runs over the selection and evaluation records are the full-scale BehaviorGPT inference
and need explicit owner approval (`approved=True`); the probe records are for small
feasibility checks and need none.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from significance import config
from significance import episodes as E
from significance import evaluate as M
from significance.data import records as R
from significance.teacher.decode import decode, requests_needed

NEEDS_APPROVAL = frozenset({"selection", "evaluation"})


class ApprovalRequired(RuntimeError):
    pass


def require_approval(record_set: str, approved: bool) -> None:
    if record_set in NEEDS_APPROVAL and not approved:
        raise ApprovalRequired(
            f"running BehaviorGPT on the {record_set} records is the full-scale inference: "
            "it needs the owner's explicit approval (pass --approved once given)"
        )


def variants_for_run(
    record_set: str, requested: Sequence[str] | None, chosen: str | None
) -> list[str]:
    """The variants a run decodes. The evaluation records run the chosen variant only
    (D16, and the request budget): it is the default there, and anything else is refused.
    Other sets run the requested variants, by default V1, V1o and V2."""
    names = list(requested) if requested else None
    if record_set == "evaluation":
        if chosen is None:
            raise ValueError("no variant chosen yet: run the selection records, then `select`")
        resolved = [chosen if n == "chosen" else n for n in (names or ["chosen"])]
        if resolved != [chosen]:
            raise ValueError(
                f"the evaluation records run the chosen variant only ({chosen}), "
                f"not {', '.join(resolved)}"
            )
        return resolved
    resolved = names or list(config.DEFAULT_VARIANTS)
    if "chosen" in resolved:
        if chosen is None:
            raise ValueError("no variant chosen yet")
        resolved = [chosen if n == "chosen" else n for n in resolved]
    unknown = [v for v in resolved if v not in config.VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}")
    return resolved


def load_set(work: Path, record_set: str) -> tuple[list[E.Episode], pd.DataFrame, pd.DataFrame]:
    """(episodes, real decisions, records) of a record set."""
    eps = E.load_episodes(E.episodes_dir(work) / f"{record_set}.jsonl")
    decisions = pd.read_parquet(E.episodes_dir(work) / f"decisions_{record_set}.parquet")
    return eps, decisions, R.load_records(work, record_set)


def predictions_path(
    work: Path, record_set: str, variant: str, empty_history: bool, limit: int | None = None
) -> Path:
    """Where a run's predictions go. A run limited to the first N records writes its own
    file (`..._firstN`), which the selection, evaluation and dataset steps never read."""
    suffix = ("_empty" if empty_history else "") + (f"_first{limit}" if limit else "")
    return work / "predictions" / record_set / f"{variant}{suffix}.parquet"


# --- estimate ------------------------------------------------------------------------


def estimate(
    episodes: Sequence[E.Episode],
    variants: Sequence[str],
    *,
    empty_history: bool = False,
    teacher=None,
) -> dict:
    """Ranking requests per variant (exact), and first-step requests already cached."""
    out = {}
    for variant in variants:
        requests = sum(requests_needed(ep.n, E.episode_k(ep), variant) for ep in episodes)
        entry = {"records": len(episodes), "requests": int(requests)}
        if teacher is not None and teacher.cache is not None:
            cached = 0
            for ep in episodes:
                k = min(E.episode_k(ep), ep.n)
                hist = E.history_for(ep, variant, k, empty_history=empty_history)
                if variant == "V3":
                    hist = [*hist, E.screen_session(ep)]
                predict = "purchase" if variant == "V3" else "view"
                for chunk in teacher.chunks(ep.presented):
                    cached += teacher.cache_key(hist, chunk, predict) in teacher.cache
            entry["first_step_requests_cached"] = cached
        out[variant] = entry
    out["total_requests"] = int(sum(v["requests"] for v in out.values()))
    return out


# --- run -----------------------------------------------------------------------------


def run_variant(
    teacher,
    episodes: Sequence[E.Episode],
    variant: str,
    *,
    empty_history: bool = False,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Decode all episodes with one variant. Returns per-observation rows and run facts."""
    started = time.time()
    before = teacher.stats.as_dict()
    done = 0

    def one(ep: E.Episode):
        return decode(teacher, ep, variant, empty_history=empty_history)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(one, episodes):
            results.append(res)
            done += 1
            if progress is not None:
                progress(done, len(episodes))
    rows = pd.DataFrame([row for res in results for row in res.rows()])
    after = teacher.stats.as_dict()
    facts = {
        "variant": variant,
        "empty_history": empty_history,
        "records": len(results),
        "observations": len(rows),
        "requests": int(sum(r.requests for r in results)),
        "teacher": {k: after[k] - before[k] for k in after},
        "records_with_missing_ids": int(sum(bool(r.missing) for r in results)),
        "history_events_mean": round(float(np.mean([r.history_events for r in results])), 2)
        if results
        else 0.0,
        "seconds": round(time.time() - started, 1),
    }
    return rows, facts


def save_predictions(rows: pd.DataFrame, facts: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    rows.to_parquet(tmp, index=False)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(facts, indent=1))


# --- choose the variant (D16) --------------------------------------------------------


def with_decisions(pred: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Predictions joined to the real decisions (one row per presented observation)."""
    keys = ["record_set", "record_id", "news_id"]
    merged = pred.merge(decisions, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError("predictions and decisions do not cover the same observations")
    return merged.reset_index(drop=True)


def best_variant(table: dict, key: Callable[[dict], float] = lambda e: e["value"]) -> str:
    """The variant with the highest `key` (mean AUC by default); ties go to the simpler."""
    order = {v: i for i, v in enumerate(config.VARIANTS)}
    return max(table, key=lambda v: (key(table[v]), -order[v]))


def choose_variant(
    predictions: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
    records: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = config.SEED,
) -> tuple[str, dict]:
    """The variant with the best mean per-record AUC against the real decisions (D16).

    Each variant's entry also holds its AUC without observations seen in an earlier
    episode (re-presented ones, plan L7), which V2 writes into the history and V1/V1o do
    not, so a reader can see whether that subset drives the choice."""
    persons = records.set_index("record_id")["user_id"]
    table = {}
    for variant, pred in predictions.items():
        merged = with_decisions(pred, decisions)
        estimates = {}
        for name, frame in (("all", merged), ("unseen", merged[~merged["seen_before"]])):
            per_record = M.ranking_table(frame.reset_index(drop=True), "confidence")
            estimates[name] = M.mean_with_ci(
                per_record["auc"].tolist(),
                persons.loc[per_record["record_id"]].tolist(),
                n_boot=n_boot,
                seed=seed,
            ).as_dict()
        table[variant] = {**estimates["all"], "without_seen_before": estimates["unseen"]}
    return best_variant(table), table
