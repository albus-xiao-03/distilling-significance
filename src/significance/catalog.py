"""MIND headlines as an Unbox AI catalog (plans/R1.md 5.2; docs/catalog-format.md).

Every observation BehaviorGPT ranks must be an item of an uploaded catalog (limit L11). Each
MIND news article becomes one text-only item:

  id                 News ID
  name               Title
  brand              null
  categories         "<Category>, <SubCategory>"
  keywords           labels of the title's entities (Wikidata names)
  search_keywords    surface forms of the title's entities
  image_url          null (text-only is allowed)
  event_type, group  "product" (required on every row)
  market             "US"
  sales_since, timestamp, frequency, price, currency: null, so no popularity or recency
  prior is baked into the catalog (plans/R1.md risk "leakage").

The parquet's file name is the catalog's name, so the slice and the full file share
`mind_small_headlines.parquet` (in different folders) and the full upload refreshes the
slice's catalog instead of taking the second slot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from behaviorgpt.resources.catalogs import CATALOG_SCHEMA, check_catalog_schema

from significance import config
from significance.data.mind import parse_entities

MARKET = "US"


def _unique(values: list[str]) -> list[str] | None:
    seen = list(dict.fromkeys(v.strip() for v in values if v and v.strip()))
    return seen or None


def entity_terms(cell: str) -> tuple[list[str] | None, list[str] | None]:
    """(keywords, search_keywords) of a Title Entities cell: entity labels and surface
    forms, deduplicated in order, or None when there are none."""
    entities = parse_entities(cell)
    labels = _unique([e.get("Label", "") for e in entities])
    surfaces = _unique([s for e in entities for s in e.get("SurfaceForms", []) or []])
    return labels, surfaces


def build_catalog(news: pd.DataFrame) -> pa.Table:
    """One catalog row per news article, typed with the SDK's CATALOG_SCHEMA."""
    if news["news_id"].duplicated().any():
        raise ValueError("news ids must be unique")
    if news["title"].str.strip().eq("").any():
        raise ValueError("every item needs a name: a news title is empty")
    terms = news["title_entities"].map(entity_terms)
    n = len(news)
    nulls = [None] * n
    columns = {
        "id": news["news_id"].tolist(),
        "name": news["title"].str.strip().tolist(),
        "brand": nulls,
        "categories": (news["category"] + ", " + news["subcategory"]).tolist(),
        "image_url": nulls,
        "event_type": ["product"] * n,
        "group": ["product"] * n,
        "sales_since": nulls,
        "timestamp": nulls,
        "frequency": nulls,
        "market": [MARKET] * n,
        "price": nulls,
        "currency": nulls,
        "search_keywords": [t[1] for t in terms],
        "keywords": [t[0] for t in terms],
    }
    schema = pa.schema(CATALOG_SCHEMA)
    return pa.table({c: pa.array(columns[c], type=schema.field(c).type) for c in CATALOG_SCHEMA})


def write_catalog(table: pa.Table, folder: Path, name: str = config.CATALOG_NAME) -> Path:
    """Write `<folder>/<name>.parquet` and check it with the SDK's schema check."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    pq.write_table(table, path)
    check_catalog_schema(path)
    return path


def catalog_slice(table: pa.Table, size: int = 300, seed: int = config.SEED) -> pa.Table:
    """A seeded random slice of the catalog, for the trial embed (plans/R1.md Phase 3)."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(table.num_rows, size=min(size, table.num_rows), replace=False))
    return table.take(pa.array(rows))
