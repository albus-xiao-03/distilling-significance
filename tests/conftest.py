"""Synthetic MIND-like fixtures. Tests never read real MIND rows."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import pytest

CATEGORIES = [("sports", "golf"), ("sports", "nfl"), ("news", "politics"), ("lifestyle", "food")]


def synthetic_news_rows(n: int = 60) -> list[list[str]]:
    rows = []
    for i in range(1, n + 1):
        cat, sub = CATEGORIES[i % len(CATEGORIES)]
        entities = (
            json.dumps([{"Label": f"Entity {i}", "SurfaceForms": [f"ent{i}"], "Type": "O"}])
            if i % 3 == 0
            else "[]"
        )
        title = f'Synthetic "{cat}" story {i}'
        rows.append([f"N{i}", cat, sub, title, f"Abstract {i}", "", entities, "[]"])
    return rows


def synthetic_behaviors(
    days: list[str], users: list[str], seed: int, start_id: int, n_news: int = 60
) -> list[list[str]]:
    """Impressions over `days` (MM/DD/YYYY): each user 1-2 impressions a day, 5-9 shown,
    1-2 clicked. Users get a fixed click history; the last user has none."""
    rng = random.Random(seed)
    histories = {
        u: [f"N{rng.randint(1, n_news)}" for _ in range(rng.randint(2, 6))] for u in users[:-1]
    }
    histories[users[-1]] = []
    rows, imp_id = [], start_id
    for day in days:
        for hour, user in enumerate(users):
            for rep in range(rng.randint(1, 2)):
                shown = rng.sample(range(1, n_news + 1), rng.randint(5, 9))
                clicks = set(rng.sample(shown, rng.randint(1, 2)))
                cell = " ".join(f"N{n}-{int(n in clicks)}" for n in shown)
                time = (
                    f"{day} {(hour % 12) + 1:02d}:{rep * 20:02d}:00 {'AM' if hour < 12 else 'PM'}"
                )
                rows.append([str(imp_id), user, time, " ".join(histories[user]), cell])
                imp_id += 1
    return rows


def write_tsv(path: Path, rows: list[list[str]]) -> Path:
    path.write_text("".join("\t".join(r) + "\n" for r in rows))
    return path


USERS = ["U1", "U2", "U3", "U4", "U5", "U6"]
TRAIN_DAYS = ["11/09/2019", "11/10/2019", "11/11/2019"]
DEV_DAYS = ["11/12/2019"]


@pytest.fixture
def mind_files(tmp_path) -> dict[str, dict[str, Path]]:
    """Synthetic behaviors.tsv / news.tsv for a train and a dev split."""
    out = {}
    news = synthetic_news_rows()
    for split, days, seed, start in (("train", TRAIN_DAYS, 1, 1), ("dev", DEV_DAYS, 2, 1000)):
        d = tmp_path / split
        d.mkdir()
        out[split] = {
            "behaviors.tsv": write_tsv(
                d / "behaviors.tsv", synthetic_behaviors(days, USERS, seed, start)
            ),
            "news.tsv": write_tsv(d / "news.tsv", news),
        }
    return out


@pytest.fixture
def mind_frames(mind_files):
    """The synthetic splits read with the package's readers: (train, dev, news)."""
    from significance.data import mind

    train = mind.read_behaviors(mind_files["train"]["behaviors.tsv"], "train")
    dev = mind.read_behaviors(mind_files["dev"]["behaviors.tsv"], "dev")
    news, _ = mind.merge_news(
        mind.read_news(mind_files["train"]["news.tsv"]),
        mind.read_news(mind_files["dev"]["news.tsv"]),
    )
    return train, dev, news


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path
