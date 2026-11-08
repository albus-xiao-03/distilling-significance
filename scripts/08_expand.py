"""Expanded scope: a 25,000-impression dataset in MIND-small's train:validation proportion.

    source .temp/env_setup/devenv.sh
    python -I scripts/08_expand.py records   # draw the extra records, build their episodes
    (label them: 04_predict.py run --set evaluation_extra / training_extra --approved, and
     evaluation_extra with --empty-history for E3a)
    python -I scripts/08_expand.py views     # assemble the dataset views (no API call)

The dataset reuses every labelled record (5,000 evaluation records from the validation
file; 1,000 selection and 20 probe records from the training file's last day) and adds:

- evaluation_extra: validation-file impressions, uniform among those not yet drawn;
- training_extra: training-file impressions with a proportional quota per day.

Episodes are built exactly as before (History clicks, then the person's training-file
impressions on days strictly before the record's day), so the representation, the
chronological cutoff and the k rule are unchanged. The profile is aggregate and, as for
the evaluation records, summarises no real decision of a dataset record before its
predictions exist. Everything is written under .temp/work/ (git-ignored).
"""

import argparse
import json

import numpy as np

from significance import config, expansion
from significance import episodes as E
from significance.data import mind, records
from significance.teacher.decode import requests_needed


def profile(eps: list[E.Episode], variant: str) -> dict:
    ks = [E.episode_k(ep) for ep in eps]
    hist = [E.count_events(E.history_for(ep, variant, k)) for ep, k in zip(eps, ks, strict=True)]
    n = np.array([ep.n for ep in eps])
    return {
        "records": len(eps),
        "with_earlier_impressions": round(
            float(np.mean([e.has_earlier_impressions for e in eps])), 4
        ),
        "history_clicks_only": round(
            float(np.mean([bool(e.previous) and not e.has_earlier_impressions for e in eps])), 4
        ),
        "no_previous_decisions": int(sum(not e.previous for e in eps)),
        "presented_mean": round(float(n.mean()), 2),
        "over_100_presented": int((n > config.MAX_IDS_PER_CALL).sum()),
        "k_mean": round(float(np.mean(ks)), 3),
        "k_equals_1": round(float(np.mean(np.array(ks) == 1)), 4),
        f"history_events_{variant}_mean": round(float(np.mean(hist)), 1),
        f"requests_{variant}": int(
            sum(requests_needed(ep.n, k, variant) for ep, k in zip(eps, ks, strict=True))
        ),
    }


def cmd_records(p) -> None:
    train, dev, _ = mind.load_tables(p.work)
    existing = {
        name: records.load_records(p.work, name) for name in ("evaluation", "selection", "probe")
    }
    extra = records.build_extra_record_sets(train, dev, existing)
    records.save_record_sets(extra, p.work)
    choice = json.loads((p.work / "selection" / "choice.json").read_text())["chosen"]
    files = records.split_counts(
        config.N_DATASET_IMPRESSIONS, {"train": len(train), "dev": len(dev)}
    )
    report = {
        "target": config.N_DATASET_IMPRESSIONS,
        "file_quotas": files,
        "variant": choice,
        "sets": {},
    }
    out_dir = E.episodes_dir(p.work)
    for name, rec in extra.items():
        eps, decisions = E.build_episodes(rec, train, dev)
        E.save_episodes(eps, out_dir / f"{name}.jsonl")
        decisions.to_parquet(out_dir / f"decisions_{name}.parquet", index=False)
        report["sets"][name] = {
            **profile(eps, choice),
            "persons": int(rec["user_id"].nunique()),
            "dates": {str(d.date()): int(c) for d, c in rec.groupby("date").size().items()},
            "observations_seen_before": round(float(decisions["seen_before"].mean()), 4),
        }
    counts = {
        "validation file": len(existing["evaluation"]) + len(extra["evaluation_extra"]),
        "training file": len(existing["selection"])
        + len(existing["probe"])
        + len(extra["training_extra"]),
    }
    report["dataset_counts"] = {**counts, "total": sum(counts.values())}
    (out_dir / "profile_extra.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_views(p) -> None:
    choice = json.loads((p.work / "selection" / "choice.json").read_text())["chosen"]
    facts = expansion.build_views(p.work, choice)
    print(json.dumps(facts, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Expanded 25,000-impression dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("records", help="draw the extra records and build their episodes")
    sub.add_parser("views", help="assemble the dataset views from the labelled sets")
    args = ap.parse_args()
    p = config.paths()
    {"records": cmd_records, "views": cmd_views}[args.cmd](p)


if __name__ == "__main__":
    main()
