"""MIND-small: download, checks, safe extraction, readers and aggregate statistics.

MIND (Microsoft News Dataset) is licensed under the Microsoft Research License Terms:
non-commercial research, no distribution. Everything read here stays under .temp/ and is
removed by scripts/delete_data.py. Downloads are untrusted data: each goes into its own new,
empty directory, extraction writes only the two expected files (by base name, so no path in
the archive can escape the target directory), and readers parse text only.

Format (msnews.github.io, "Introduction to MIND"):
  behaviors.tsv  Impression ID, User ID, Time ("MM/DD/YYYY HH:MM:SS AM/PM"), History (clicked
                 news ids before the impression, oldest first), Impressions ("N123-1 N456-0",
                 1 = clicked, shown in shuffled order)
  news.tsv       News ID, Category, SubCategory, Title, Abstract, URL, Title Entities,
                 Abstract Entities (JSON lists)
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from significance import config

WANTED_FILES = ("behaviors.tsv", "news.tsv")
NEWS_COLUMNS = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)
BEHAVIOR_COLUMNS = ("impression_id", "user_id", "time", "history", "impressions")
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
MAX_MEMBER_BYTES = 2_000_000_000


# --- download and extraction ---------------------------------------------------------


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: dict[str, Any]) -> bool:
    """Whether `path` has the expected size and sha256."""
    return (
        path.is_file() and path.stat().st_size == spec["size"] and sha256_of(path) == spec["sha256"]
    )


def _is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def download_split(split: str, raw_dir: Path, token: str, *, downloader=None) -> Path:
    """Download one MIND-small zip into its own new, empty directory and verify it.

    Idempotent: a verified zip already in place is reused. A non-empty directory without
    a verified zip is refused rather than overwritten. `downloader` (tests) replaces
    huggingface_hub.hf_hub_download.
    """
    spec = config.MIND_FILES[split]
    target = raw_dir / f"mind_small_{split}"
    zip_path = target / spec["filename"]
    if zip_path.is_file() and verify_file(zip_path, spec):
        return zip_path
    if target.exists() and not _is_empty_dir(target):
        raise RuntimeError(
            f"{target} is not empty but holds no verified {spec['filename']}: remove it first"
        )
    target.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader
    path = Path(
        downloader(
            config.MIND_REPO,
            spec["filename"],
            repo_type="dataset",
            revision=config.MIND_REVISION,
            token=token,
            local_dir=target,
        )
    )
    if path.resolve() != zip_path.resolve() or not verify_file(path, spec):
        raise RuntimeError(f"{spec['filename']}: size or sha256 does not match the pinned file")
    return zip_path


def extract_split(zip_path: Path, dest: Path) -> dict[str, Path]:
    """Extract behaviors.tsv and news.tsv (and nothing else) into `dest`, a new, empty
    directory, flattened to their base names. Unsafe member paths are refused.

    Idempotent: a marker records the zip's sha256 and the files' sizes.
    """
    marker = dest / "extracted.json"
    zip_sha = sha256_of(zip_path)
    if marker.is_file():
        info = json.loads(marker.read_text())
        files = {name: dest / name for name in WANTED_FILES}
        if info.get("zip_sha256") == zip_sha and all(
            files[n].is_file() and files[n].stat().st_size == info["sizes"][n] for n in files
        ):
            return files
    if dest.exists() and not _is_empty_dir(dest):
        raise RuntimeError(f"{dest} is not empty: remove it before extracting again")
    dest.mkdir(parents=True, exist_ok=True)

    members: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
            if name.name in WANTED_FILES:
                if name.name in members:
                    raise ValueError(f"archive holds more than one {name.name}")
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ValueError(f"{name.name} is implausibly large ({info.file_size} B)")
                members[name.name] = info
        missing = set(WANTED_FILES) - set(members)
        if missing:
            raise ValueError(f"archive lacks {sorted(missing)}")
        for base, info in members.items():
            with zf.open(info) as src, (dest / base).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    files = {name: dest / name for name in WANTED_FILES}
    marker.write_text(
        json.dumps(
            {"zip_sha256": zip_sha, "sizes": {n: p.stat().st_size for n, p in files.items()}}
        )
    )
    return files


# --- readers -------------------------------------------------------------------------


def read_news(path: Path) -> pd.DataFrame:
    """news.tsv as a string table (no quoting: titles keep their quote characters)."""
    news = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(NEWS_COLUMNS),
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error",
    )
    if news["news_id"].eq("").any():
        raise ValueError(f"{path.name}: empty news id")
    return news


def merge_news(*tables: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per news id over several news.tsv tables (first occurrence kept), and the
    number of ids whose rows disagree between tables."""
    both = pd.concat(tables, ignore_index=True)
    dup = both[both.duplicated("news_id", keep=False)]
    conflicts = int(
        dup.groupby("news_id")[["title", "category", "subcategory"]]
        .nunique()
        .gt(1)
        .any(axis=1)
        .sum()
    )
    return both.drop_duplicates("news_id", keep="first").reset_index(drop=True), conflicts


def parse_entities(text: str) -> list[dict[str, Any]]:
    """A Title/Abstract Entities cell as a list of dicts ([] when empty)."""
    text = (text or "").strip()
    if not text:
        return []
    value = json.loads(text)
    return value if isinstance(value, list) else []


def _parse_impressions(cell: str, where: str) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    """'N1-1 N2-0 ...' -> (presented ids, labels, duplicates removed). A news id listed
    twice is kept once, at its first position, clicked if either entry was clicked."""
    order: list[str] = []
    labels: dict[str, int] = {}
    duplicates = 0
    for token in cell.split():
        news_id, sep, label = token.rpartition("-")
        if not sep or not news_id or label not in ("0", "1"):
            raise ValueError(f"{where}: malformed impression entry {token!r}")
        if news_id in labels:
            duplicates += 1
            labels[news_id] = max(labels[news_id], int(label))
        else:
            order.append(news_id)
            labels[news_id] = int(label)
    return tuple(order), tuple(labels[n] for n in order), duplicates


def read_behaviors(path: Path, split: str) -> pd.DataFrame:
    """behaviors.tsv as one row per impression:

    impression_id, user_id, time, date, split, history (tuple of clicked ids, oldest
    first), presented (tuple of shown ids, deduplicated, MIND's shuffled order), clicked
    (tuple of 0/1 aligned with presented), n_presented, n_clicked, n_duplicates.
    """
    rows = []
    with path.open(newline="") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(BEHAVIOR_COLUMNS):
                raise ValueError(f"{path.name}:{line_no}: expected 5 fields, got {len(parts)}")
            imp_id, user_id, time, history, impressions = parts
            presented, clicked, dups = _parse_impressions(impressions, f"{path.name}:{line_no}")
            if not presented:
                raise ValueError(f"{path.name}:{line_no}: impression shows nothing")
            rows.append(
                (int(imp_id), user_id, time, tuple(history.split()), presented, clicked, dups)
            )
    beh = pd.DataFrame(
        rows,
        columns=[
            "impression_id",
            "user_id",
            "time",
            "history",
            "presented",
            "clicked",
            "n_duplicates",
        ],
    )
    beh["time"] = pd.to_datetime(beh["time"], format=TIME_FORMAT)
    beh["date"] = beh["time"].dt.normalize()
    beh["split"] = split
    beh["n_presented"] = beh["presented"].map(len).astype("int64")
    beh["n_clicked"] = beh["clicked"].map(sum).astype("int64")
    if beh["impression_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate impression ids")
    return beh


# --- aggregate statistics (plans/R1.md 3.5) ------------------------------------------


def _dist(values: pd.Series) -> dict[str, float]:
    v = values.to_numpy(dtype=float)
    return {
        "mean": round(float(v.mean()), 3),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def describe(train: pd.DataFrame, dev: pd.DataFrame, news: pd.DataFrame) -> dict[str, Any]:
    """Aggregate facts only (no rows): the Phase 1 checklist of plans/R1.md 3.5, plus what
    the episode builder relies on."""
    news_ids = set(news["news_id"])
    out: dict[str, Any] = {
        "news_count": len(news_ids),
        "news_fits_one_catalog": len(news_ids) <= 100_000,
    }
    last_train_day = train["date"].max()
    train_users_before_last = set(train.loc[train["date"] < last_train_day, "user_id"])
    train_users = set(train["user_id"])
    for name, beh in (("train", train), ("dev", dev)):
        presented_ids = {n for p in beh["presented"] for n in p}
        history_ids = {n for h in beh["history"] for n in h}
        per_user_hist = beh.groupby("user_id")["history"].nunique()
        out[name] = {
            "impressions": len(beh),
            "users": int(beh["user_id"].nunique()),
            "dates": sorted(str(d.date()) for d in beh["date"].unique()),
            "presented_per_impression": _dist(beh["n_presented"]),
            "clicks_per_impression": _dist(beh["n_clicked"]),
            "impressions_without_click": int((beh["n_clicked"] == 0).sum()),
            "impressions_all_clicked": int((beh["n_clicked"] == beh["n_presented"]).sum()),
            "impressions_over_100_presented": int((beh["n_presented"] > 100).sum()),
            "click_rate_pooled": round(float(beh["n_clicked"].sum() / beh["n_presented"].sum()), 5),
            "history_length": _dist(beh["history"].map(len)),
            "impressions_with_empty_history": int((beh["history"].map(len) == 0).sum()),
            "users_with_varying_history": int((per_user_hist > 1).sum()),
            "duplicate_entries_removed": int(beh["n_duplicates"].sum()),
            "presented_ids_missing_from_news": len(presented_ids - news_ids),
            "history_ids_missing_from_news": len(history_ids - news_ids),
        }
    last_day = train[train["date"] == last_train_day]
    out["selection_pool"] = {
        "date": str(last_train_day.date()),
        "impressions": len(last_day),
        "share_with_earlier_train_impressions": round(
            float(last_day["user_id"].isin(train_users_before_last).mean()), 4
        ),
    }
    out["evaluation_pool"] = {
        "impressions": len(dev),
        "share_with_train_impressions": round(float(dev["user_id"].isin(train_users).mean()), 4),
    }
    both = pd.concat([train[["user_id", "history"]], dev[["user_id", "history"]]])
    out["users_with_history_differing_between_files"] = int(
        (both.drop_duplicates().groupby("user_id").size() > 1).sum()
    )
    return out
