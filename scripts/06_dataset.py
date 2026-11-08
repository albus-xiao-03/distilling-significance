"""Write the pseudo-label dataset into the dataset folder (plans/R1.md 7.2; expanded scope).

    source .temp/env_setup/devenv.sh
    python -I scripts/08_expand.py views   # first: assemble the labelled record sets
    python -I scripts/06_dataset.py

Reads the `dataset` view of the chosen variant's predictions (25,000 impressions) and writes,
into the dataset folder directly under the project:

  train.parquet        impressions of MIND-small's training file (17,053)
  validation.parquet   impressions of MIND-small's validation file (7,947)
  metadata.json        scope, splits, schema, record-id ranges, provenance, API usage,
                       checksums

Each file has the six columns of section 7.2 only. The folder is git-ignored: never commit,
push or redistribute it (README "Rights notice").
"""

import json
import sys
from datetime import date

import pandas as pd

from significance import config, dataset, runner
from significance.data import records

ORIGINS = {
    "evaluation": "validation file; the first dataset's 5,000 evaluation records, ids unchanged",
    "evaluation_extra": "validation file; added in the expansion",
    "selection": "training file, 2019-11-14; their real decisions chose the representation "
    "(D16), so they are left out of the evaluation",
    "probe": "training file, 2019-11-14; the small API checks",
    "training_extra": "training file, 2019-11-09 to 2019-11-14; added in the expansion",
}
LABEL_SOURCES = ("evaluation", "evaluation_extra", "selection", "probe", "training_extra")


def api_usage(p, variant: str) -> dict:
    """API calls behind the labels (one run per source set) and behind evaluation-only runs."""

    def facts(record_set, name):
        path = p.work / "predictions" / record_set / f"{name}.json"
        return json.loads(path.read_text())["teacher"] if path.is_file() else None

    labels = {s: facts(s, variant) for s in LABEL_SOURCES}
    other = {
        f"{s}/{name}": facts(s, name)
        for s, name in (
            ("evaluation", f"{variant}_empty"),
            ("evaluation_extra", f"{variant}_empty"),
            ("selection", "V1"),
            ("selection", "V2"),
        )
    }
    total = {k: sum(v[k] for v in labels.values()) for k in ("api_calls", "cache_hits")}
    return {
        "label_runs": labels,
        "label_runs_total": total,
        "evaluation_only_runs": {k: v for k, v in other.items() if v is not None},
        "note": "probe responses date from the API checks and were re-keyed in the cache; "
        "every label can be re-decoded from the cache with no API call",
    }


def main() -> None:
    p = config.paths()
    choice = p.work / "selection" / "choice.json"
    if not choice.is_file():
        sys.exit("no variant chosen yet (04_predict.py select)")
    variant = json.loads(choice.read_text())["chosen"]
    path = runner.predictions_path(p.work, "dataset", variant, False)
    if not path.is_file():
        sys.exit("no dataset view yet: run scripts/08_expand.py views first")
    predictions = pd.read_parquet(path)
    recs = records.load_records(p.work, "dataset")
    if len(recs) != config.N_DATASET_IMPRESSIONS or predictions["record_id"].nunique() != len(recs):
        sys.exit(f"the dataset view must cover {config.N_DATASET_IMPRESSIONS} records")
    table = dataset.build_output(predictions, variant, "dataset")
    split_of = recs.set_index("record_id")["split"].map({"train": "train", "dev": "validation"})
    paths = dataset.write_dataset(table, split_of.to_dict(), p.dataset)

    ids = pd.read_parquet(p.work / "dataset" / "id_map.parquet")
    ranges = []
    for origin, part in ids.groupby("origin", sort=False):
        ranges.append(
            {
                "record_ids": f"{part['dataset_id'].min()}-{part['dataset_id'].max()}",
                "records": len(part),
                "file": part["split"].iloc[0],
                "origin": ORIGINS[origin],
                "in_evaluation": origin != "selection",
            }
        )
    meta = dataset.build_metadata(
        paths,
        ranges,
        variant,
        api_usage(p, variant),
        created=date.today().isoformat(),
        extra={"documentation": ["DATACARD.md", "reports/R1_expanded_results.md"]},
    )
    (p.dataset / dataset.METADATA_NAME).write_text(json.dumps(meta, indent=1))
    for split, facts in meta["splits"].items():
        print(
            f"{split:10s}: {facts['records']:6d} records, {facts['rows']:7d} rows -> "
            f"{(p.dataset / facts['file']).relative_to(p.root)} (git-ignored)"
        )
    print(f"metadata -> {(p.dataset / dataset.METADATA_NAME).relative_to(p.root)}")


if __name__ == "__main__":
    main()
