"""Record sampler (plans/R1.md 6.5 and 7.1; decisions D16, D17, N4).

A record is one decision episode: one MIND impression. Three disjoint record sets:

- evaluation: N_EVALUATION_RECORDS impressions of the MIND-small validation file (the last
  day of week 5). Their pseudo-labels form the output dataset; their real decisions are
  used for evaluation only.
- selection: N_SELECTION_RECORDS impressions of the training file's last day. They choose
  the representation variant, so the choice never sees evaluation data.
- probe: N_PROBE_RECORDS further impressions of the training file's last day, disjoint from
  the selection records, for the small API feasibility checks only.

Expanded scope (owner, 2026-11-08): the dataset holds N_DATASET_IMPRESSIONS impressions,
split between the training and validation files in proportion to their sizes. Two more
sets complete it, reusing every labelled record above:

- evaluation_extra: more validation-file impressions, drawn uniformly from the ones not yet
  drawn, so the validation part stays a uniform sample of the file;
- training_extra: training-file impressions with a proportional quota per day (the
  existing last-day selection and probe records count towards their day's quota), so the
  training part is spread over the file's days like the file itself.

`record_id` (C3) is the record's position in the seeded random draw, not MIND's impression
id; the mapping to impression ids stays under .temp/work/. It is not anonymous: with MIND and
this code the draw can be redone, and a record's presented set identifies its impression, so
anything keyed by it is MIND-derived data and is never distributed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from significance import config

RECORD_COLUMNS = ["record_set", "record_id", "impression_id", "user_id", "time", "date", "split"]
RECORD_SETS = ("evaluation", "selection", "probe", "evaluation_extra", "training_extra")
EXTRA_SETS = ("evaluation_extra", "training_extra")


def sample_records(pool: pd.DataFrame, n: int, seed: int, record_set: str) -> pd.DataFrame:
    """`n` impressions of `pool`, drawn uniformly without replacement with `seed`."""
    if n > len(pool):
        raise ValueError(f"{record_set}: asked for {n} records from a pool of {len(pool)}")
    rng = np.random.default_rng(seed)
    picked = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
    rec = picked[["impression_id", "user_id", "time", "date", "split"]].reset_index(drop=True)
    rec.insert(0, "record_id", np.arange(n, dtype="int64"))
    rec.insert(0, "record_set", record_set)
    return rec[RECORD_COLUMNS]


def build_record_sets(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    *,
    n_evaluation: int = config.N_EVALUATION_RECORDS,
    n_selection: int = config.N_SELECTION_RECORDS,
    n_probe: int = config.N_PROBE_RECORDS,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """The evaluation, selection and probe records (disjoint by construction)."""
    last_day = train["date"].max()
    last_day_pool = train[train["date"] == last_day]
    evaluation = sample_records(dev, n_evaluation, seed, "evaluation")
    selection = sample_records(last_day_pool, n_selection, seed + 1, "selection")
    rest = last_day_pool[~last_day_pool["impression_id"].isin(selection["impression_id"])]
    probe = sample_records(rest, n_probe, seed + 2, "probe")
    return {"evaluation": evaluation, "selection": selection, "probe": probe}


def split_counts(n_total: int, sizes: Mapping) -> dict:
    """`n_total` apportioned over groups in proportion to `sizes` (largest remainder; ties
    in the remainder go to the group listed first)."""
    total = sum(sizes.values())
    if total <= 0 or n_total < 0:
        raise ValueError("sizes must sum to a positive number and n_total be >= 0")
    exact = {k: n_total * v / total for k, v in sizes.items()}
    counts = {k: math.floor(x) for k, x in exact.items()}
    order = sorted(exact, key=lambda k: -(exact[k] - counts[k]))
    for k in order[: n_total - sum(counts.values())]:
        counts[k] += 1
    return counts


def build_extra_record_sets(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    existing: Mapping[str, pd.DataFrame],
    *,
    n_total: int = config.N_DATASET_IMPRESSIONS,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """The evaluation_extra and training_extra records: with the existing evaluation,
    selection and probe records they make `n_total` impressions, split between the files
    in proportion to their sizes, the training part with a proportional quota per day."""
    files = split_counts(n_total, {"train": len(train), "dev": len(dev)})
    taken = pd.concat([existing[k] for k in ("evaluation", "selection", "probe")])
    have_dev = existing["evaluation"]
    if files["dev"] < len(have_dev):
        raise ValueError(
            f"{len(have_dev)} existing validation records exceed the quota {files['dev']}"
        )
    pool = dev[~dev["impression_id"].isin(have_dev["impression_id"])]
    evaluation_extra = sample_records(
        pool, files["dev"] - len(have_dev), seed + 3, "evaluation_extra"
    )

    days = train.groupby("date").size().sort_index()
    quotas = split_counts(files["train"], days.to_dict())
    have_train = taken[taken["split"] == "train"]
    parts = []
    for i, (day, quota) in enumerate(quotas.items()):
        have = have_train[have_train["date"] == day]
        need = quota - len(have)
        if need < 0:
            raise ValueError(f"{day.date()}: {len(have)} existing records exceed the quota {quota}")
        day_pool = train[
            (train["date"] == day) & ~train["impression_id"].isin(have["impression_id"])
        ]
        rng = np.random.default_rng(seed + 10 + i)
        parts.append(day_pool.iloc[rng.choice(len(day_pool), size=need, replace=False)])
    drawn = pd.concat(parts)
    # a seeded shuffle, so a record's position (its record_id) says nothing about its day
    drawn = drawn.iloc[np.random.default_rng(seed + 4).permutation(len(drawn))]
    training_extra = sample_records(drawn, len(drawn), seed + 5, "training_extra")
    return {"evaluation_extra": evaluation_extra, "training_extra": training_extra}


def records_path(work: Path, record_set: str) -> Path:
    return work / "records" / f"{record_set}.parquet"


def save_record_sets(sets: dict[str, pd.DataFrame], work: Path) -> dict[str, Path]:
    out = {}
    for name, frame in sets.items():
        path = records_path(work, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        out[name] = path
    return out


def load_records(work: Path, record_set: str) -> pd.DataFrame:
    if record_set not in RECORD_SETS:
        raise ValueError(f"unknown record set {record_set!r}")
    return pd.read_parquet(records_path(work, record_set))
