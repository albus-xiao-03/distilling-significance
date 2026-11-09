"""A private copy of the dataset on the owner's Hugging Face account (owner, 2026-11-09).

The dataset is MIND-derived data under the Microsoft Research License Terms (no
distribution), so it is uploaded only to a PRIVATE dataset repository of the owner, and
only these files of the dataset folder go up:

  README.md            the dataset card (rendered here from metadata.json; no data)
  train.parquet        the training-file impressions
  validation.parquet   the validation-file impressions
  metadata.json        scope, schema, record-id ranges, method, API usage, checksums

`check_private` refuses a repository that is missing or public; `verify` compares every
uploaded file with the local one by its checksum.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

UPLOAD_FILES = ("README.md", "train.parquet", "validation.parquet", "metadata.json")
CODE_URL = "https://github.com/albus-xiao-03/distilling-significance"
LICENSE_LINK = "https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf"


class PublishError(RuntimeError):
    pass


def check_private(api, repo_id: str):
    """The repository's info, if it exists and is private; otherwise PublishError."""
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except Exception as exc:
        raise PublishError(f"cannot read {repo_id}: {type(exc).__name__}") from exc
    if not getattr(info, "private", False):
        raise PublishError(f"{repo_id} is not private: the MIND licence forbids distribution")
    return info


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload(api, folder: Path, repo_id: str, message: str, files: Sequence[str] = UPLOAD_FILES):
    """Upload exactly `files` of `folder` in one commit."""
    missing = [f for f in files if not (folder / f).is_file()]
    if missing:
        raise PublishError(f"missing in {folder}: {', '.join(missing)}")
    return api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="dataset",
        allow_patterns=list(files),
        commit_message=message,
    )


def verify(api, folder: Path, repo_id: str, files: Sequence[str] = UPLOAD_FILES) -> dict:
    """file -> True if the repository holds the same bytes (LFS sha256, or git blob id)."""
    remote = {f.path: f for f in api.get_paths_info(repo_id, list(files), repo_type="dataset")}
    result = {}
    for name in files:
        info = remote.get(name)
        if info is None:
            result[name] = False
        elif getattr(info, "lfs", None) is not None:
            result[name] = info.lfs.sha256 == sha256_of(folder / name)
        else:
            result[name] = info.blob_id == git_blob_sha1(folder / name)
    return result


def render_card(meta: dict) -> str:
    """The Hugging Face dataset card (README.md), from metadata.json. No data in it."""
    splits = meta["splits"]
    columns = "\n".join(
        f"| `{name}` | {col['type']} | {col['meaning']} |" for name, col in meta["columns"].items()
    )
    ranges = "\n".join(
        f"| {r['record_ids']} | {r['records']:,} | {r['file']} | {r['origin']} | "
        f"{'yes' if r['in_evaluation'] else 'no'} |"
        for r in meta["record_id_ranges"]
    )
    method = meta["method"]
    split_rows = "\n".join(
        f"| {name} | `{name}.parquet` | {splits[name]['records']:,} | {splits[name]['rows']:,} | "
        f"{splits[name]['attended_rows']:,} |"
        for name in ("train", "validation")
    )
    return f"""---
pretty_name: MIND attention pseudo-labels (BehaviorGPT)
license: other
license_name: microsoft-research-license-terms
license_link: {LICENSE_LINK}
language:
- en
tags:
- pseudo-labels
- news
- attention
- mind
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
---

# MIND attention pseudo-labels

> **Private copy. Do not share, mirror or make this repository public.** The data is derived
> from MIND, licensed under the [Microsoft Research License Terms]({LICENSE_LINK}):
> non-commercial research only, **no distribution**. The labels are outputs of BehaviorGPT
> (Unbox AI), subject to Unbox AI's terms. Every record can be traced back to a MIND
> impression.

**BehaviorGPT's predicted human attention decisions** on MIND-small news impressions: for
each headline shown to a person, whether BehaviorGPT, used unchanged, predicts that the person
opens it. The predictions are pseudo-labels; the real decisions are not included.

Code, method and full documentation: [{CODE_URL}]({CODE_URL}) (see `DATACARD.md` and
`reports/R1_expanded_results.md`).

## Files

| Split | File | Records (impressions) | Rows (headlines shown) | Labelled attended |
|---|---|---|---|---|
{split_rows}

`metadata.json` holds the scope, schema, record-id ranges, method, API usage and sha256
checksums. The splits follow MIND-small's files: the impressions are drawn in proportion to
the files' sizes, and the training part has a proportional quota per day.

## Columns

| Column | Type | Meaning |
|---|---|---|
{columns}

No user ids, timestamps, histories, headline text or real clicks are included.

## Record ids

| record_id | Records | MIND file | Origin | In the evaluation |
|---|---|---|---|---|
{ranges}

## How the labels were made

- **Model:** {method["model"]}.
- **History:** {method["history"]}.
- **Representation:** `{method["variant"]}`, previous clicks written as views, then an order.
- **Decoding:** {method["decoding"]}.
- **k rule:** {method["k_rule"]}.

## Rights and deletion

- **Licence.** MIND belongs to Microsoft (Microsoft Research License Terms; no distribution).
- **Private storage only.** This repository is a private copy kept by the project owner; it is
  never made public or shared.
- **Deletion.** It is deleted together with the project's other data: on a rights holder's
  request, if the terms turn out to disallow this use, and at the end of the project (see the
  deletion term in the code repository's README).
"""
