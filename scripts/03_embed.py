"""Phase 3: upload the MIND headline catalog to a private Unbox AI catalog and check it.

    source .temp/env_setup/devenv.sh
    python -I scripts/03_embed.py slice   # trial: the 300-row slice
    python -I scripts/03_embed.py full    # all headlines; same name, so it refreshes the slice
    python -I scripts/03_embed.py check   # status, coverage and a similar_items spot check

The upload is approved under decision N3 and the catalog is deleted under D18
(scripts/delete_data.py). Both files are named mind_small_headlines.parquet, and the file
name is the catalog's name, so the project uses one catalog slot. Items are embedded with
reference_time = the start of the validation day, 2019-11-15 (plans/R1.md 5.2); the catalog
has no sales or timestamps, so nothing is measured from it.

The spot check prints aggregates only: how often an item's similar items share its MIND
category and subcategory, against how often random items would. No headline is printed.
The catalog id and job details are kept in .temp/work/catalog_state.json.
"""

import argparse
import json
import random
import time
from collections import Counter
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq
from behaviorgpt import UnboxAIClient

from significance import config

REFERENCE_TIME = datetime(2019, 11, 15, tzinfo=UTC)
SPOT_CHECK_ITEMS = 20


def state_path(p):
    return p.work / "catalog_state.json"


def load_state(p) -> dict:
    path = state_path(p)
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(p, state: dict) -> None:
    path = state_path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1, default=str))


def owned_catalog(client):
    """The project's own catalog: owned by this key and named after the catalog file."""
    mine = [
        c
        for c in client.list_catalogs()
        if c.owned and c.catalog_name.removesuffix(".parquet") == config.CATALOG_NAME
    ]
    if len(mine) > 1:
        raise RuntimeError(f"{len(mine)} owned catalogs are named {config.CATALOG_NAME}")
    return mine[0] if mine else None


def embed(client, p, which: str) -> None:
    path = p.catalog / which / f"{config.CATALOG_NAME}.parquet"
    rows = pq.read_metadata(path).num_rows
    print(f"embedding {which}: {rows} items from {path.relative_to(p.root)}")
    started = time.time()
    job = client.embed(path, wait=True, timeout=3600.0, reference_time=REFERENCE_TIME)
    status = client.job_status(job.job_id)
    state = load_state(p)
    state.update(
        {
            "catalog_id": job.catalog_id,
            "catalog_name": job.catalog_name,
            "last_embed": {
                "which": which,
                "items": rows,
                "job_id": job.job_id,
                "status": status.status,
                "items_rejected": status.items_rejected.count if status.items_rejected else 0,
                "seconds": round(time.time() - started, 1),
                "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        }
    )
    save_state(p, state)
    print(json.dumps(state, indent=1))


def check(client, p) -> None:
    cat = owned_catalog(client)
    if cat is None:
        raise SystemExit("no owned catalog named mind_small_headlines: run `full` first")
    print(f"owned catalog: status {cat.status}")
    state = load_state(p)
    if state.get("catalog_id") != cat.catalog_id:
        raise SystemExit("catalog id differs from .temp/work/catalog_state.json")

    news = pd.read_parquet(
        p.work / "mind" / "news.parquet", columns=["news_id", "category", "subcategory"]
    )
    info = news.set_index("news_id")
    rng = random.Random(config.SEED)
    sample = rng.sample(list(info.index), SPOT_CHECK_ITEMS)
    same_cat = same_sub = total = missing = 0
    for news_id in sample:
        res = client.similar_items(news_id, limit=10, catalog_id=cat.catalog_id)
        hits = [it.id for it in res.items if it.id != news_id]
        missing += sum(h not in info.index for h in hits)
        hits = [h for h in hits if h in info.index]
        total += len(hits)
        same_cat += sum(info.loc[h, "category"] == info.loc[news_id, "category"] for h in hits)
        same_sub += sum(
            info.loc[h, "subcategory"] == info.loc[news_id, "subcategory"] for h in hits
        )
    cat_share = Counter(info["category"])
    sub_share = Counter(info["subcategory"])
    n = len(info)
    base_cat = sum(cat_share[info.loc[x, "category"]] / n for x in sample) / len(sample)
    base_sub = sum(sub_share[info.loc[x, "subcategory"]] / n for x in sample) / len(sample)
    result = {
        "similar_items_calls": len(sample),
        "neighbours_checked": total,
        "neighbours_not_in_mind": missing,
        "same_category": round(same_cat / total, 3),
        "same_category_if_random": round(base_cat, 3),
        "same_subcategory": round(same_sub / total, 3),
        "same_subcategory_if_random": round(base_sub, 3),
    }
    state["spot_check"] = result
    save_state(p, state)
    print(json.dumps(result, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["slice", "full", "check"])
    args = ap.parse_args()
    p = config.paths()
    config.load_env()
    config.require_secret("UNBOXAI_API_KEY")
    client = UnboxAIClient(market=config.MARKET)
    if args.step == "check":
        check(client, p)
    else:
        embed(client, p, args.step)


if __name__ == "__main__":
    main()
