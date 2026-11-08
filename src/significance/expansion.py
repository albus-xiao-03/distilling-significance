"""The expanded dataset as views over the labelled record sets (owner, 2026-11-08).

The dataset's 25,000 records come from five record sets, each labelled once by its own run
(every response is cached; nothing is decoded twice):

  validation file  evaluation (5,000), evaluation_extra (2,947)
  training file    selection (1,000), probe (20), training_extra (16,033)

Every record gets a dataset record_id. The evaluation records keep theirs (0-4,999, as in
the first, 5,000-record dataset), and the other sets follow in the order above. The views
re-key records, episodes, real decisions and predictions to those ids, so the existing
evaluation scripts run on them unchanged:

  dataset             all records: the output
  dataset_eval        all but the selection records, whose real decisions chose the
                      variant (D16): E1/E2, baselines, strata and E4
  dataset_validation  the validation-file records, with the empty-history run: E3a
  dataset_training    the training-file records but the selection records
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from significance import episodes as E
from significance import runner
from significance.data import records as R

#: (record set, MIND file) in dataset-id order; the evaluation records come first.
SOURCES = (
    ("evaluation", "validation"),
    ("evaluation_extra", "validation"),
    ("selection", "train"),
    ("probe", "train"),
    ("training_extra", "train"),
)
VIEWS: dict[str, tuple[str, ...]] = {
    "dataset": ("evaluation", "evaluation_extra", "selection", "probe", "training_extra"),
    "dataset_eval": ("evaluation", "evaluation_extra", "probe", "training_extra"),
    "dataset_validation": ("evaluation", "evaluation_extra"),
    "dataset_training": ("probe", "training_extra"),
}
#: Views evaluated with the empty-history run as well (E3a).
EMPTY_HISTORY_VIEWS = ("dataset_validation",)
KEYS = ["record_set", "record_id"]


def id_map(records_by_set: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """record_set, record_id -> dataset_id, MIND file ('train' / 'validation') and origin."""
    parts, offset = [], 0
    for name, split in SOURCES:
        rec = records_by_set[name]
        ids = sorted(rec["record_id"].tolist())
        if ids != list(range(len(rec))):
            raise ValueError(f"{name}: record ids must be 0..{len(rec) - 1}")
        parts.append(
            pd.DataFrame(
                {
                    "record_set": name,
                    "record_id": ids,
                    "dataset_id": [offset + i for i in ids],
                    "split": split,
                    "origin": name,
                }
            )
        )
        offset += len(rec)
    return pd.concat(parts, ignore_index=True)


def rekey(frame: pd.DataFrame, ids: pd.DataFrame, view: str) -> pd.DataFrame:
    """`frame` with record_set = view and record_id = the dataset id."""
    merged = frame.merge(ids[[*KEYS, "dataset_id"]], on=KEYS, how="left", validate="many_to_one")
    if merged["dataset_id"].isna().any():
        raise ValueError("a row's record is missing from the id map")
    merged["record_set"] = view
    merged["record_id"] = merged.pop("dataset_id").astype("int64")
    return merged[list(frame.columns)]


def rekey_episodes(episodes: Sequence[E.Episode], ids: pd.DataFrame, view: str) -> list[E.Episode]:
    lookup = {(r.record_set, r.record_id): r.dataset_id for r in ids.itertuples(index=False)}
    return [
        replace(ep, record_set=view, record_id=int(lookup[(ep.record_set, ep.record_id)]))
        for ep in episodes
    ]


def _load_source(work: Path, name: str, variant: str) -> dict:
    eps, decisions, rec = runner.load_set(work, name)
    pred = pd.read_parquet(runner.predictions_path(work, name, variant, False))
    if pred["record_id"].nunique() != len(rec) or set(pred["variant"]) != {variant}:
        raise ValueError(f"{name}: {variant} predictions do not cover all {len(rec)} records")
    if pred["empty_history"].any():
        raise ValueError(f"{name}: empty-history rows among the labels")
    return {"episodes": eps, "decisions": decisions, "records": rec, "predictions": pred}


def build_views(work: Path, variant: str) -> dict[str, dict]:
    """Write the four views under .temp/work/ (records, episodes, decisions, predictions)
    and the id map; return aggregate facts per view."""
    sources = {name: _load_source(work, name, variant) for name, _ in SOURCES}
    ids = id_map({name: s["records"] for name, s in sources.items()})
    (work / "dataset").mkdir(parents=True, exist_ok=True)
    ids.to_parquet(work / "dataset" / "id_map.parquet", index=False)
    facts = {}
    for view, members in VIEWS.items():
        rec = pd.concat([rekey(sources[m]["records"], ids, view) for m in members])
        eps = [e for m in members for e in rekey_episodes(sources[m]["episodes"], ids, view)]
        dec = pd.concat([rekey(sources[m]["decisions"], ids, view) for m in members])
        pred = pd.concat([rekey(sources[m]["predictions"], ids, view) for m in members])
        rec.sort_values("record_id").to_parquet(R.records_path(work, view), index=False)
        E.save_episodes(
            sorted(eps, key=lambda e: e.record_id), E.episodes_dir(work) / f"{view}.jsonl"
        )
        dec.to_parquet(E.episodes_dir(work) / f"decisions_{view}.parquet", index=False)
        view_facts = {
            "records": len(rec),
            "observations": len(pred),
            "sources": {m: len(sources[m]["records"]) for m in members},
            "split": {
                k: int(v)
                for k, v in rec["split"]
                .map({"train": "train", "dev": "validation"})
                .value_counts()
                .items()
            },
        }
        runner.save_predictions(
            pred,
            {**view_facts, "variant": variant, "view_of": list(members)},
            runner.predictions_path(work, view, variant, False),
        )
        if view in EMPTY_HISTORY_VIEWS:
            empty = pd.concat(
                [
                    rekey(
                        pd.read_parquet(runner.predictions_path(work, m, variant, True)), ids, view
                    )
                    for m in members
                ]
            )
            if empty["record_id"].nunique() != len(rec):
                raise ValueError(f"{view}: the empty-history run does not cover every record")
            runner.save_predictions(
                empty,
                {**view_facts, "variant": variant, "empty_history": True},
                runner.predictions_path(work, view, variant, True),
            )
        facts[view] = view_facts
    (work / "dataset" / "views.json").write_text(json.dumps(facts, indent=1))
    return facts
