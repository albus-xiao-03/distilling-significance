"""Phase 1: get MIND-small, check it, describe it and draw the records.

    source .temp/env_setup/devenv.sh
    python -I scripts/01_prepare.py

1. Download MINDsmall_train.zip and MINDsmall_dev.zip from the gated Hugging Face copy
   (HF_TOKEN from .env, never printed) into their own new directories under
   .temp/data/raw/, at a pinned revision, and check size and sha256.
2. Extract only behaviors.tsv and news.tsv into new directories.
3. Parse them and store the tables under .temp/work/mind/.
4. Print and store aggregate statistics (plans/R1.md 3.5): no rows, no ids.
5. Draw the evaluation, selection and probe records into .temp/work/records/.

Everything it writes is git-ignored and removed by scripts/delete_data.py.
"""

import json

from significance import config
from significance.data import mind, records


def main() -> None:
    p = config.paths()
    names = config.load_env()
    print(".env variables:", ", ".join(names) or "none", "(values not shown)")
    token = config.require_secret("HF_TOKEN")

    tables = {}
    for split in ("train", "dev"):
        zip_path = mind.download_split(split, p.raw, token)
        print(f"{split}: {zip_path.name} verified (size and sha256)")
        files = mind.extract_split(zip_path, p.raw / f"mind_small_{split}_tsv")
        tables[split] = {
            "behaviors": mind.read_behaviors(files["behaviors.tsv"], split),
            "news": mind.read_news(files["news.tsv"]),
        }
    news, conflicts = mind.merge_news(tables["train"]["news"], tables["dev"]["news"])
    train, dev = tables["train"]["behaviors"], tables["dev"]["behaviors"]
    out = mind.save_tables(p.work, train, dev, news)
    print(f"tables stored under {out.relative_to(p.root)}")

    stats = mind.describe(train, dev, news)
    stats["news_rows_conflicting_between_files"] = conflicts
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    sets = records.build_record_sets(train, dev)
    paths = records.save_record_sets(sets, p.work)
    for name, frame in sets.items():
        print(
            f"records {name:10s}: {len(frame):5d} impressions, {frame['user_id'].nunique():5d} "
            f"persons -> {paths[name].relative_to(p.root)}"
        )


if __name__ == "__main__":
    main()
