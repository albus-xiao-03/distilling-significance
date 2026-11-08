"""Phase 2: the catalog file and the decision episodes.

    source .temp/env_setup/devenv.sh
    python -I scripts/02_build.py

1. Catalog (plans/R1.md 5.2): .temp/data/catalog/full/mind_small_headlines.parquet with
   every MIND-small news article, and a 300-row seeded slice in
   .temp/data/catalog/slice/ for the trial embed; both pass the SDK's schema check.
2. Episodes (5.1, 5.3) for the evaluation, selection and probe records in
   .temp/work/episodes/<set>.jsonl, and their real decisions apart in
   .temp/work/episodes/decisions_<set>.parquet.
3. An aggregate profile of each record set (no rows, no ids): observations, k, history
   events per variant after the cap K, observations seen before. Real decisions are
   summarised for the selection and probe sets only; nothing about the evaluation
   records' decisions is looked at before their predictions exist.
"""

import json

import numpy as np
import pandas as pd

from significance import catalog, config
from significance import episodes as E
from significance.data import mind, records


def _dist(values) -> dict:
    v = np.asarray(list(values), dtype=float)
    return {
        "mean": round(float(v.mean()), 3),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
    }


def profile(name: str, eps: list[E.Episode], decisions: pd.DataFrame) -> dict:
    ks = [E.episode_k(ep) for ep in eps]
    out = {
        "records": len(eps),
        "with_earlier_impressions": round(
            float(np.mean([e.has_earlier_impressions for e in eps])), 4
        ),
        "with_history_clicks": round(float(np.mean([e.history_clicks > 0 for e in eps])), 4),
        "with_no_previous_decisions": int(sum(not e.previous for e in eps)),
        "presented": _dist(e.n for e in eps),
        "over_100_presented": int(sum(e.n > config.MAX_IDS_PER_CALL for e in eps)),
        "k": _dist(ks),
        "k_counts": {
            str(k): int(c) for k, c in zip(*np.unique(ks, return_counts=True), strict=True)
        },
        "observations_seen_before": round(float(decisions["seen_before"].mean()), 4),
        "observations_attended_before": round(float(decisions["attended_before"].mean()), 4),
        "history_events_after_cap": {},
    }
    for variant in config.VARIANTS:
        sizes, capped = [], 0
        for ep, k in zip(eps, ks, strict=True):
            hist = E.history_for(ep, variant, k)
            full = E.previous_sessions(ep, variant, 10**9)
            sizes.append(E.count_events(hist))
            capped += E.count_events(hist) < E.count_events(full)
        out["history_events_after_cap"][variant] = {**_dist(sizes), "records_capped": capped}
    if name != "evaluation":
        clicks = decisions.groupby("record_id")["attended"].sum()
        true = np.array([clicks[e.record_id] for e in eps])
        out["k_rule_diagnostic"] = {
            "mean_true_clicks": round(float(true.mean()), 3),
            "mean_abs_k_error": round(float(np.mean(np.abs(np.array(ks) - true))), 3),
            "k_equals_true": round(float(np.mean(np.array(ks) == true)), 4),
        }
    return out


def main() -> None:
    p = config.paths()
    train, dev, news = mind.load_tables(p.work)

    table = catalog.build_catalog(news)
    full = catalog.write_catalog(table, p.catalog / "full")
    trial = catalog.catalog_slice(table)
    part = catalog.write_catalog(trial, p.catalog / "slice")
    print(f"catalog: {table.num_rows} items -> {full.relative_to(p.root)}")
    print(f"catalog slice: {trial.num_rows} items -> {part.relative_to(p.root)}")

    out_dir = E.episodes_dir(p.work)
    summary = {}
    for name in records.RECORD_SETS:
        rec = records.load_records(p.work, name)
        eps, decisions = E.build_episodes(rec, train, dev)
        E.save_episodes(eps, out_dir / f"{name}.jsonl")
        decisions.to_parquet(out_dir / f"decisions_{name}.parquet", index=False)
        summary[name] = profile(name, eps, decisions)
        print(f"episodes {name}: {len(eps)} -> {(out_dir / f'{name}.jsonl').relative_to(p.root)}")
    (out_dir / "profile.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
