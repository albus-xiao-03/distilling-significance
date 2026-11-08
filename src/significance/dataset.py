"""The output dataset: item identifiers and created labels only (plans/R1.md 7.2; D15, C3).

outputs/mind_attention_pseudolabels.parquet (git-ignored, never committed or pushed), one
row per presented observation of the 5,000 evaluation records:

  record_id            created: opaque number of the decision episode
  news_id              original: the item identifier, the only original field
  predicted_attention  created: BehaviorGPT's predicted decision (the pseudo-label)
  attention_rank       created: 1 = BehaviorGPT's first predicted action
  confidence           created: 1 - (attention_rank - 1) / n
  variant              created: provenance (the representation used)

Not in it: person ids, times, histories, titles, categories and the real decisions.
`validate_output` enforces this, and that every record's labels are consistent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from significance import config

OUTPUT_NAME = "mind_attention_pseudolabels.parquet"
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


def output_path(outputs: Path) -> Path:
    return outputs / OUTPUT_NAME


def build_output(predictions: pd.DataFrame, variant: str) -> pa.Table:
    """The output table from the chosen variant's predictions on the evaluation records."""
    if variant not in config.VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if set(predictions["record_set"]) != {"evaluation"}:
        raise ValueError("the output holds the evaluation records only")
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


def write_output(table: pa.Table, outputs: Path) -> Path:
    validate_output(table)
    path = output_path(outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path
