"""The deletion term of README.md (plans/R1.md 7.4; decisions D10, D18).

Deletes the project's MIND-derived data and its own Unbox AI catalog:

- local: .temp/data/ (downloads, extracted files, HF cache, catalog files), .temp/work/
  (records, episodes, real decisions, predictions, evaluation tables), .temp/cache/ (the
  BehaviorGPT response cache), dataset/ (the label dataset) and outputs/ (where the first,
  superseded dataset was written);
- remote: the catalog that this API key owns and that is named mind_small_headlines.

Shared catalogs and catalogs with any other name are never touched. Only those fixed
folders inside the project are deleted, a symlink is removed as a link and never
followed, and nothing happens without `confirm=True` (the default is a dry run).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from significance import config

LOCAL_TARGETS = (".temp/data", ".temp/work", ".temp/cache", "dataset", "outputs")


@dataclass
class Report:
    local: list[str] = field(default_factory=list)
    catalogs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: bool = False

    def lines(self) -> list[str]:
        verb = "deleted" if self.deleted else "would delete"
        out = [f"{verb}: {p}" for p in self.local]
        out += [f"{verb} catalog: {c}" for c in self.catalogs]
        out += [f"skipped: {s}" for s in self.skipped]
        if not (self.local or self.catalogs):
            out.append("nothing to delete")
        return out


def local_targets(root: Path) -> list[Path]:
    """The existing deletion targets inside `root` (symlinks included, never followed)."""
    root = root.resolve()
    found = []
    for rel in LOCAL_TARGETS:
        path = root / rel
        if not (path.exists() or path.is_symlink()):
            continue
        parent = path.parent.resolve()
        if not parent.is_relative_to(root):
            raise RuntimeError(f"{rel}: its folder resolves outside the project")
        found.append(path)
    return found


def is_project_catalog(summary) -> bool:
    """Owned by this key and named after the project's catalog file."""
    name = str(summary.catalog_name).removesuffix(".parquet")
    return bool(summary.owned) and name == config.CATALOG_NAME


def project_catalogs(client) -> list:
    return [c for c in client.list_catalogs() if is_project_catalog(c)]


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def run(root: Path, client, *, confirm: bool = False) -> Report:
    """Delete (or, by default, list) the local targets and the project's catalog.
    `client` may be None, e.g. without an API key: the catalog step is then skipped."""
    report = Report(deleted=confirm)
    targets = local_targets(root)
    report.local = [str(p.relative_to(root)) for p in targets]
    catalogs = []
    if client is None:
        report.skipped.append(
            "Unbox AI catalog: no API client (set UNBOXAI_API_KEY in .env, or delete the "
            f"owned catalog named {config.CATALOG_NAME} by hand)"
        )
    else:
        catalogs = project_catalogs(client)
        report.catalogs = [f"{c.catalog_name} ({c.catalog_id})" for c in catalogs]
    if confirm:
        for c in catalogs:  # the catalog first: its id is also recorded under .temp/work
            client.delete_catalog(c.catalog_id)
        for path in targets:
            _remove(path)
    return report
