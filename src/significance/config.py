"""Paths, fixed settings and the .env loader for R1 (plans/R1.md).

Secrets (UNBOXAI_API_KEY, HF_TOKEN) are read from the git-ignored .env into this process
only. Their values are never printed, logged or written anywhere; only their names are
reported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- fixed settings ------------------------------------------------------------------

#: Hugging Face copy of MIND (gated; Microsoft Research License Terms) and the files used.
MIND_REPO = "yjw1029/MIND"
#: Repository commit the files below were verified at (HF API, 2026-11-08).
MIND_REVISION = "0f871bdfd1ac4029324a48722ec6d52d714ebf27"
MIND_FILES = {
    "train": {
        "filename": "MINDsmall_train.zip",
        "size": 52_953_372,
        "sha256": "a966e5138ad103376e9817e02395719bf1c62ec56e6e98c30d46fbb991a7fafa",
    },
    "dev": {
        "filename": "MINDsmall_dev.zip",
        "size": 30_946_172,
        "sha256": "b315cde1c9b9d45008b5a7c4b2e1f87647659f09f74892ae3899c0005d5d6155",
    },
}

#: The project's own Unbox AI catalog. The parquet file name is the catalog's name.
CATALOG_NAME = "mind_small_headlines"
MARKET = "us"

#: Ids per ranking call: the API takes at most 100 (verified in Phase 0).
MAX_IDS_PER_CALL = 100
#: History cap K in events, the current session included. Phase 0 measured that BehaviorGPT
#: reads about the last 256-270 events of a history and drops older ones, so histories are
#: capped at 256 by our own rule (whole recent sessions kept) rather than cut by the server.
MAX_HISTORY_EVENTS = 256

#: Records (plans/R1.md 6.5, 7.1; decisions D16, D17, N4).
N_EVALUATION_RECORDS = 5_000
N_SELECTION_RECORDS = 1_000
N_PROBE_RECORDS = 20
SEED = 20_261_108

#: Expanded scope (owner, 2026-11-08): the dataset holds 25,000 impressions in total, split
#: between MIND-small's training and validation files in proportion to their sizes
#: (156,965 : 73,152 -> 17,053 : 7,947). It reuses the labelled evaluation, selection and
#: probe records and adds new records to reach those counts.
N_DATASET_IMPRESSIONS = 25_000
#: The dataset folder directly under the project folder (git-ignored, deletion term).
DATASET_FOLDER = "dataset"

#: Representation variants (plans/R1.md 5.4). V3 failed its Phase 0 go/no-go probe
#: (predict="purchase" did not follow the person's history within the screen), so it is
#: not run by default (plans/R1.md L6).
VARIANTS = ("V1", "V1o", "V2", "V3")
DEFAULT_VARIANTS = ("V1", "V1o", "V2")

#: Output dataset columns, in order (plans/R1.md 7.2; decisions D15, C3).
OUTPUT_COLUMNS = (
    "record_id",
    "news_id",
    "predicted_attention",
    "attention_rank",
    "confidence",
    "variant",
)


# --- paths ---------------------------------------------------------------------------


def project_root() -> Path:
    """The project folder: $DS_ROOT (set by .temp/env_setup/devenv.sh), else found from
    this file of the editable install."""
    env = os.environ.get("DS_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "plans").is_dir():
            return parent
    raise RuntimeError("project root not found: source .temp/env_setup/devenv.sh first")


@dataclass(frozen=True)
class Paths:
    """Where everything lives. Data, caches and evaluation tables are git-ignored."""

    root: Path

    @property
    def temp(self) -> Path:
        return self.root / ".temp"

    @property
    def data(self) -> Path:
        """Downloads, extracted files, HF cache and catalog files (deletion term)."""
        return self.temp / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def catalog(self) -> Path:
        return self.data / "catalog"

    @property
    def work(self) -> Path:
        """Records, episodes, real decisions, predictions, evaluation (deletion term)."""
        return self.temp / "work"

    @property
    def cache(self) -> Path:
        """The BehaviorGPT response cache (deletion term)."""
        return self.temp / "cache"

    @property
    def outputs(self) -> Path:
        """Where the first, 5,000-record dataset was written; superseded by `dataset`."""
        return self.root / "outputs"

    @property
    def dataset(self) -> Path:
        """The pseudo-label dataset folder (git-ignored, never pushed; deletion term)."""
        return self.root / DATASET_FOLDER

    @property
    def reports(self) -> Path:
        """Committed aggregate results only."""
        return self.root / "reports"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"


def paths() -> Paths:
    return Paths(project_root())


# --- .env ----------------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE pairs of a .env file. Blank lines and comments are skipped, an `export `
    prefix is allowed, and matching quotes around a value are removed."""
    pairs: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if name:
            pairs[name] = value
    return pairs


def load_env(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Put the variables of .env into os.environ and return their names (never values).
    Variables already set in the environment win unless `override`."""
    path = path or paths().env_file
    if not path.is_file():
        return []
    pairs = read_env_file(path)
    for name, value in pairs.items():
        if override or name not in os.environ:
            os.environ[name] = value
    return sorted(pairs)


def require_secret(name: str) -> str:
    """The value of a secret from the environment (after `load_env`). A missing secret
    raises an error that names it; the value itself is never part of any message."""
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set: add it to the project's .env")
    return value
