"""MIND reader: download handling, safe extraction, parsing and statistics (synthetic data)."""

import hashlib

import pandas as pd
import pytest

from conftest import make_zip
from significance import config
from significance.data import mind


# --- parsing -------------------------------------------------------------------------
def test_read_behaviors_parses_fields(tmp_path):
    p = tmp_path / "behaviors.tsv"
    p.write_text(
        "7\tU1\t11/15/2019 10:22:32 AM\tN3 N1\tN9-0 N8-1 N7-0\n"
        "8\tU2\t11/15/2019 01:05:00 PM\t\tN5-1\n"
    )
    beh = mind.read_behaviors(p, "dev")
    first = beh.iloc[0]
    assert first["impression_id"] == 7 and first["user_id"] == "U1"
    assert first["time"] == pd.Timestamp("2019-11-15 10:22:32")
    assert first["date"] == pd.Timestamp("2019-11-15")
    assert first["history"] == ("N3", "N1")
    assert first["presented"] == ("N9", "N8", "N7") and first["clicked"] == (0, 1, 0)
    assert (first["n_presented"], first["n_clicked"]) == (3, 1)
    second = beh.iloc[1]
    assert second["history"] == () and second["time"].hour == 13
    assert set(beh["split"]) == {"dev"}


def test_duplicate_entries_are_merged_clicked_if_any(tmp_path):
    p = tmp_path / "behaviors.tsv"
    p.write_text("1\tU1\t11/15/2019 10:00:00 AM\t\tN1-0 N2-0 N1-1\n")
    row = mind.read_behaviors(p, "dev").iloc[0]
    assert row["presented"] == ("N1", "N2") and row["clicked"] == (1, 0)
    assert row["n_duplicates"] == 1


@pytest.mark.parametrize("cell", ["N1-2", "N1", "-1", "N1-x"])
def test_malformed_impression_entries_raise(tmp_path, cell):
    p = tmp_path / "behaviors.tsv"
    p.write_text(f"1\tU1\t11/15/2019 10:00:00 AM\t\t{cell}\n")
    with pytest.raises(ValueError, match="malformed"):
        mind.read_behaviors(p, "dev")


def test_wrong_field_count_and_duplicate_ids_raise(tmp_path):
    p = tmp_path / "behaviors.tsv"
    p.write_text("1\tU1\t11/15/2019 10:00:00 AM\tN1-1\n")
    with pytest.raises(ValueError, match="5 fields"):
        mind.read_behaviors(p, "dev")
    p.write_text("1\tU1\t11/15/2019 10:00:00 AM\t\tN1-1\n1\tU2\t11/15/2019 10:00:00 AM\t\tN2-1\n")
    with pytest.raises(ValueError, match="duplicate impression"):
        mind.read_behaviors(p, "dev")


def test_read_news_keeps_quotes_and_parses_entities(mind_files):
    news = mind.read_news(mind_files["train"]["news.tsv"])
    assert list(news.columns) == list(mind.NEWS_COLUMNS)
    assert news.loc[0, "title"].startswith('Synthetic "')
    ents = mind.parse_entities(news.loc[news["news_id"] == "N3", "title_entities"].item())
    assert ents[0]["Label"] == "Entity 3"
    assert mind.parse_entities("") == [] and mind.parse_entities("[]") == []


def test_merge_news_keeps_first_and_counts_conflicts():
    a = pd.DataFrame(
        {
            "news_id": ["N1", "N2"],
            "title": ["t1", "t2"],
            "category": ["c", "c"],
            "subcategory": ["s", "s"],
        }
    )
    b = pd.DataFrame(
        {
            "news_id": ["N2", "N3"],
            "title": ["t2 changed", "t3"],
            "category": ["c", "c"],
            "subcategory": ["s", "s"],
        }
    )
    merged, conflicts = mind.merge_news(a, b)
    assert list(merged["news_id"]) == ["N1", "N2", "N3"]
    assert merged.loc[merged["news_id"] == "N2", "title"].item() == "t2"
    assert conflicts == 1


# --- extraction ----------------------------------------------------------------------
def test_extract_split_writes_only_wanted_files_flattened(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {
            "MINDsmall_dev/behaviors.tsv": b"b",
            "MINDsmall_dev/news.tsv": b"n",
            "MINDsmall_dev/entity_embedding.vec": b"e",
        },
    )
    dest = tmp_path / "out"
    files = mind.extract_split(z, dest)
    assert files["behaviors.tsv"].read_bytes() == b"b"
    assert sorted(p.name for p in dest.iterdir()) == ["behaviors.tsv", "extracted.json", "news.tsv"]
    assert mind.extract_split(z, dest) == files  # idempotent


@pytest.mark.parametrize("bad", ["../news.tsv", "/abs/news.tsv", "a\\..\\news.tsv"])
def test_extract_split_refuses_unsafe_paths(tmp_path, bad):
    z = make_zip(tmp_path / "a.zip", {bad: b"x", "behaviors.tsv": b"b"})
    with pytest.raises(ValueError, match="unsafe"):
        mind.extract_split(z, tmp_path / "out")


def test_extract_split_needs_both_files_and_an_empty_target(tmp_path):
    z = make_zip(tmp_path / "a.zip", {"behaviors.tsv": b"b"})
    with pytest.raises(ValueError, match="lacks"):
        mind.extract_split(z, tmp_path / "out")
    full = make_zip(tmp_path / "b.zip", {"behaviors.tsv": b"b", "news.tsv": b"n"})
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "other").write_text("x")
    with pytest.raises(RuntimeError, match="not empty"):
        mind.extract_split(full, busy)


# --- download ------------------------------------------------------------------------
def _fake_spec(monkeypatch, payload: bytes):
    spec = {
        "filename": "MINDsmall_dev.zip",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setitem(config.MIND_FILES, "dev", spec)
    return spec


def test_download_split_uses_a_new_dir_verifies_and_reuses(tmp_path, monkeypatch):
    payload = b"zip-bytes"
    _fake_spec(monkeypatch, payload)
    calls = []

    def fake_download(repo, filename, *, repo_type, revision, token, local_dir):
        calls.append((repo, filename, repo_type, revision, token))
        out = local_dir / filename
        out.write_bytes(payload)
        return str(out)

    path = mind.download_split("dev", tmp_path, "tok", downloader=fake_download)
    assert path == tmp_path / "mind_small_dev" / "MINDsmall_dev.zip"
    assert calls == [
        (config.MIND_REPO, "MINDsmall_dev.zip", "dataset", config.MIND_REVISION, "tok")
    ]
    mind.download_split("dev", tmp_path, "tok", downloader=fake_download)
    assert len(calls) == 1  # the verified file is reused


def test_download_split_rejects_bad_checksum_and_busy_dirs(tmp_path, monkeypatch):
    _fake_spec(monkeypatch, b"expected")

    def wrong(repo, filename, *, repo_type, revision, token, local_dir):
        (local_dir / filename).write_bytes(b"tampered")
        return str(local_dir / filename)

    with pytest.raises(RuntimeError, match="sha256"):
        mind.download_split("dev", tmp_path / "a", "tok", downloader=wrong)
    busy = tmp_path / "b" / "mind_small_dev"
    busy.mkdir(parents=True)
    (busy / "stray").write_text("x")
    with pytest.raises(RuntimeError, match="not empty"):
        mind.download_split("dev", tmp_path / "b", "tok", downloader=wrong)


# --- statistics ----------------------------------------------------------------------
def test_describe_reports_aggregates(mind_frames):
    train, dev, news = mind_frames
    stats = mind.describe(train, dev, news)
    assert stats["news_count"] == 60 and stats["news_fits_one_catalog"]
    assert stats["train"]["impressions"] == len(train)
    assert stats["train"]["dates"] == ["2019-11-09", "2019-11-10", "2019-11-11"]
    assert stats["dev"]["impressions_without_click"] == 0
    assert stats["train"]["presented_ids_missing_from_news"] == 0
    assert stats["selection_pool"]["date"] == "2019-11-11"
    assert stats["selection_pool"]["share_with_earlier_train_impressions"] == 1.0
    assert stats["evaluation_pool"]["share_with_train_impressions"] == 1.0
    assert stats["dev"]["impressions_with_empty_history"] == int((dev["user_id"] == "U6").sum())
