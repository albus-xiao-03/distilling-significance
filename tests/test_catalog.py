"""Catalog builder: SDK schema, column mapping, slice (synthetic news)."""

import pandas as pd
import pyarrow.parquet as pq
import pytest
from behaviorgpt.resources.catalogs import CATALOG_SCHEMA

from significance import catalog, config


def test_catalog_matches_sdk_schema_and_mapping(mind_frames, tmp_path):
    _, _, news = mind_frames
    table = catalog.build_catalog(news)
    assert table.column_names == list(CATALOG_SCHEMA)
    assert table.num_rows == len(news)
    path = catalog.write_catalog(table, tmp_path)
    assert path.name == f"{config.CATALOG_NAME}.parquet"
    back = pq.read_table(path).to_pandas()
    row = back.set_index("id").loc["N3"]
    src = news.set_index("news_id").loc["N3"]
    assert row["name"] == src["title"]
    assert row["categories"] == f"{src['category']}, {src['subcategory']}"
    assert list(row["keywords"]) == ["Entity 3"] and list(row["search_keywords"]) == ["ent3"]
    assert row["event_type"] == "product" and row["group"] == "product"
    assert row["market"] == "US"
    for col in ("brand", "image_url", "sales_since", "timestamp", "frequency", "price"):
        assert back[col].isna().all(), col
    assert back.set_index("id").loc["N1", "keywords"] is None  # no entities -> null


def test_entity_terms_dedupes_and_handles_empty():
    cell = (
        '[{"Label": "A", "SurfaceForms": ["a", "a "]}, {"Label": "A", "SurfaceForms": ["b"]},'
        ' {"Label": "", "SurfaceForms": []}]'
    )
    assert catalog.entity_terms(cell) == (["A"], ["a", "b"])
    assert catalog.entity_terms("") == (None, None)


def test_catalog_rejects_duplicate_ids_and_empty_titles():
    base = {"category": ["c", "c"], "subcategory": ["s", "s"], "title_entities": ["[]", "[]"]}
    with pytest.raises(ValueError, match="unique"):
        catalog.build_catalog(pd.DataFrame({"news_id": ["N1", "N1"], "title": ["a", "b"], **base}))
    with pytest.raises(ValueError, match="name"):
        catalog.build_catalog(pd.DataFrame({"news_id": ["N1", "N2"], "title": ["a", " "], **base}))


def test_catalog_slice_is_seeded_and_sized(mind_frames):
    _, _, news = mind_frames
    table = catalog.build_catalog(news)
    a = catalog.catalog_slice(table, size=10, seed=3)
    b = catalog.catalog_slice(table, size=10, seed=3)
    assert a.num_rows == 10 and a.column("id").to_pylist() == b.column("id").to_pylist()
    assert catalog.catalog_slice(table, size=10_000).num_rows == table.num_rows
