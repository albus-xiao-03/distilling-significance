"""Record sampler (plans/R1.md 6.5 and 7.1; decisions D16, D17, N4).

A record is one decision episode: one MIND impression. Three disjoint record sets:

- evaluation: N_EVALUATION_RECORDS impressions of the MIND-small validation file (the last
  day of week 5). Their pseudo-labels form the output dataset; their real decisions are
  used for evaluation only.
- selection: N_SELECTION_RECORDS impressions of the training file's last day. They choose
  the representation variant, so the choice never sees evaluation data.
- probe: N_PROBE_RECORDS further impressions of the training file's last day, disjoint from
  the selection records, for the small API feasibility checks only.

`record_id` is opaque (C3): the record's position in the seeded random draw, unrelated to
MIND's impression id, user or time. The mapping to impression ids stays under .temp/work/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from significance import config

RECORD_COLUMNS = ["record_set", "record_id", "impression_id", "user_id", "time", "date", "split"]
RECORD_SETS = ("evaluation", "selection", "probe")


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
