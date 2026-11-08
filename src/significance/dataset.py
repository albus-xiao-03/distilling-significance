"""The output dataset: item identifiers and created labels only (plans/R1.md 7.2; D15, C3).

The dataset folder (config.DATASET_FOLDER, directly under the project; git-ignored, never
committed or pushed) holds one parquet file per MIND-small file the impressions come from,
with one row per presented observation and exactly these columns:

  record_id            created: number of the decision episode, unique across both files
  news_id              original: the item identifier, the only original field
  predicted_attention  created: BehaviorGPT's predicted decision (the pseudo-label)
  attention_rank       created: 1 = BehaviorGPT's first predicted action
  confidence           created: 1 - (attention_rank - 1) / n
  variant              created: provenance (the representation used)

  train.parquet        impressions of MIND-small's training file
  validation.parquet   impressions of MIND-small's validation file
  metadata.json        scope, splits, schema, record-id ranges, provenance, checksums

Not in it: person ids, times, histories, titles, categories and the real decisions.
`validate_output` enforces this for each file, and that every record's labels are
consistent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from significance import config

SCHEMA = pa.schema(
    [
        ("record_id", pa.int64()),
        ("news_id", pa.string()),
        ("predicted_attention", pa.int8()),
        ("attention_rank", pa.int32()),
        ("confidence", pa.float64()),
        ("variant", pa.string()),
    ]
)
assert tuple(SCHEMA.names) == config.OUTPUT_COLUMNS
SPLIT_FILES = {"train": "train.parquet", "validation": "validation.parquet"}
METADATA_NAME = "metadata.json"
COLUMN_DOCS = {
    "record_id": "created: number of the decision episode (one MIND impression), unique "
    "across the dataset; not MIND's impression id",
    "news_id": "original: MIND news id of a headline shown in the impression",
    "predicted_attention": "created: 1 if BehaviorGPT predicts the person opened the "
    "headline, else 0 (the pseudo-label)",
    "attention_rank": "created: the step at which BehaviorGPT chose the headline as the "
    "person's next action (1 = first); unchosen headlines follow by their last scores",
    "confidence": "created: 1 - (attention_rank - 1) / n, rank-based",
    "variant": "created: the representation that produced the labels",
}


def build_output(predictions: pd.DataFrame, variant: str, record_set: str = "dataset") -> pa.Table:
    """The output table from the chosen variant's predictions on `record_set`."""
    if variant not in config.VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if set(predictions["record_set"]) != {record_set}:
        raise ValueError(f"the output holds the {record_set} records only")
    if predictions["empty_history"].any():
        raise ValueError("empty-history predictions (E3a) are not pseudo-labels")
    if set(predictions["variant"]) != {variant}:
        raise ValueError("predictions mix variants or are not the chosen one")
    out = (
        predictions[list(config.OUTPUT_COLUMNS)]
        .sort_values(["record_id", "attention_rank"])
        .reset_index(drop=True)
    )
    table = pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)
    validate_output(table)
    return table


def validate_output(table: pa.Table) -> None:
    """Exactly the six columns with their types, and consistent labels per record:
    ranks 1..n, the top k ranks attended (k >= 1), confidence 1 - (rank - 1)/n."""
    if table.schema.remove_metadata() != SCHEMA:
        raise ValueError(f"output schema must be exactly {SCHEMA}, got {table.schema}")
    df = table.to_pandas()
    if df.duplicated(["record_id", "news_id"]).any():
        raise ValueError("an observation appears twice in a record")
    if not df["predicted_attention"].isin([0, 1]).all():
        raise ValueError("predicted_attention must be 0 or 1")
    for record_id, part in df.groupby("record_id"):
        n = len(part)
        ranks = part["attention_rank"].sort_values().tolist()
        if ranks != list(range(1, n + 1)):
            raise ValueError(f"record {record_id}: ranks are not 1..{n}")
        k = int(part["predicted_attention"].sum())
        if k < 1 or not (part["predicted_attention"] == (part["attention_rank"] <= k)).all():
            raise ValueError(f"record {record_id}: attended observations are not the top {k}")
        expected = 1.0 - (part["attention_rank"] - 1) / n
        if not (part["confidence"] - expected).abs().lt(1e-9).all():
            raise ValueError(f"record {record_id}: confidence does not follow the rank")


def _write(table: pa.Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def write_dataset(table: pa.Table, split_of: Mapping[int, str], folder: Path) -> dict[str, Path]:
    """One validated file per MIND file ('train' / 'validation') in `folder`."""
    validate_output(table)
    df = table.to_pandas()
    splits = df["record_id"].map(split_of)
    if splits.isna().any() or not set(splits) <= set(SPLIT_FILES):
        raise ValueError("every record needs a split: 'train' or 'validation'")
    paths = {}
    for split, name in SPLIT_FILES.items():
        part = df[splits == split].reset_index(drop=True)
        part_table = pa.Table.from_pandas(part, schema=SCHEMA, preserve_index=False)
        validate_output(part_table)
        paths[split] = _write(part_table, folder / name)
    return paths


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def file_facts(path: Path) -> dict:
    table = pq.read_table(path)
    df = table.to_pandas()
    return {
        "file": path.name,
        "records": int(df["record_id"].nunique()),
        "rows": len(df),
        "attended_rows": int(df["predicted_attention"].sum()),
        "record_id_min": int(df["record_id"].min()),
        "record_id_max": int(df["record_id"].max()),
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def build_metadata(
    paths: Mapping[str, Path],
    id_ranges: list[dict],
    variant: str,
    api_usage: dict,
    created: str,
    extra: Mapping | None = None,
) -> dict:
    """metadata.json: scope, splits, schema, record-id ranges, provenance and rights."""
    splits = {split: file_facts(path) for split, path in paths.items()}
    ids = [
        set(pq.read_table(p, columns=["record_id"]).column(0).to_pylist()) for p in paths.values()
    ]
    if len(ids) == 2 and ids[0] & ids[1]:
        raise ValueError("a record_id appears in both files")
    return {
        "name": "mind_attention_pseudolabels",
        "description": "BehaviorGPT's predicted human attention decisions on MIND-small "
        "news impressions (which headlines a person opens), used as pseudo-labels.",
        "created": created,
        "records": sum(s["records"] for s in splits.values()),
        "rows": sum(s["rows"] for s in splits.values()),
        "splits": splits,
        "split_rule": "impressions drawn in proportion to MIND-small's file sizes (training "
        "156,965 : validation 73,152); the training part with a proportional quota per day",
        "columns": {
            name: {"type": str(SCHEMA.field(name).type), "meaning": COLUMN_DOCS[name]}
            for name in SCHEMA.names
        },
        "record_id_ranges": id_ranges,
        "method": {
            "model": "BehaviorGPT (Unbox AI), unchanged: no fine-tuning or training",
            "variant": variant,
            "decoding": "next-action decoding over the presented headlines, k steps",
            "k_rule": "k = max(1, round(a * n)), a = the person's attention rate in earlier "
            "impressions; k = 1 without them",
            "history": "History clicks, then the person's training-file impressions on days "
            "strictly before the impression's day; capped at 256 events",
        },
        "api_usage": api_usage,
        "rights": "MIND is licensed under the Microsoft Research License Terms "
        "(non-commercial research, no distribution). The labels are BehaviorGPT outputs "
        "(Unbox AI terms). Records can be traced back to MIND impressions, so this folder is "
        "MIND-derived data: never commit, push or redistribute it. See README.md and "
        "DATACARD.md.",
        **dict(extra or {}),
    }
