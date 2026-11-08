"""Phase 5: the predicted attention decisions against the real ones (plans/R1.md 8).

    source .temp/env_setup/devenv.sh
    python -I scripts/05_evaluate.py --set probe
    python -I scripts/05_evaluate.py --set selection
    python -I scripts/05_evaluate.py --set evaluation

Evaluates every BehaviorGPT prediction file of the set (variants, empty-history runs) and
the four baselines (random, popularity, recency, content), each with the same k rule:
E1 ranking metrics, E2 decision metrics, the k error, the oracle-k diagnostic, AUC by
stratum (earlier impressions / history clicks only / no previous decisions), AUC without
re-presented observations (L7), and paired AUC differences (E3; BehaviorGPT against the
baselines). 95 % intervals resample persons.

The evaluation records are refused until the chosen variant's predictions exist, so no
evaluation result is seen before the pseudo-labels are fixed. Output, aggregates only:
.temp/work/eval/<set>/results.json and results.md.
"""

import argparse
import json
import sys

import pandas as pd

from significance import baselines as B
from significance import config, runner
from significance import episodes as E
from significance import evaluate as M
from significance import experiment as X
from significance.data import mind


def fmt(est: dict) -> str:
    lo, hi = est["ci95"]
    return f"{est['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def markdown(results: dict) -> str:
    lines = [
        f"# Evaluation: {results['set']} records",
        "",
        f"{results['records']} records, {results['persons']} persons. Strata: "
        + ", ".join(f"{k} {v}" for k, v in results["strata_counts"].items())
        + ". 95 % person-bootstrap intervals.",
        "",
        "| Predictor | AUC | MRR | nDCG@5 | nDCG@10 | F1 | kappa | oracle-k precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results["predictors"].items():
        e1, e2 = r["E1"], r["E2"]
        lines.append(
            f"| {name} | {fmt(e1['auc'])} | {e1['mrr']['value']:.3f} | "
            f"{e1['ndcg5']['value']:.3f} | {e1['ndcg10']['value']:.3f} | "
            f"{fmt(e2['f1'])} | {e2['kappa']['value']:.3f} | "
            f"{r['oracle_k_precision']['value']:.3f} |"
        )
    lines += [
        "",
        "## AUC by stratum",
        "",
        "| Predictor | " + " | ".join(results["strata_counts"]) + " |",
        "|---|" + "---|" * len(results["strata_counts"]),
    ]
    for name, r in results["predictors"].items():
        cells = [
            fmt(r["E1_by_stratum"][s]) if s in r["E1_by_stratum"] else "-"
            for s in results["strata_counts"]
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired AUC differences (E3)", "", "| Comparison | Difference |", "|---|---|"]
    for name, d in results["comparisons"].items():
        lines.append(f"| {name} | {fmt(d)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate predicted attention decisions (E1-E3)")
    ap.add_argument("--set", default="evaluation", choices=["evaluation", "selection", "probe"])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    p = config.paths()
    eps, decisions, recs = runner.load_set(p.work, args.set)
    decisions = decisions.reset_index(drop=True)
    persons = recs.set_index("record_id")["user_id"]
    labels = X.strata(eps)
    k_by_record = {(ep.record_set, ep.record_id): min(E.episode_k(ep), ep.n) for ep in eps}

    predictors: dict[str, pd.DataFrame] = {}
    for variant in config.VARIANTS:
        for empty in (False, True):
            path = runner.predictions_path(p.work, args.set, variant, empty)
            if path.is_file():
                pred = pd.read_parquet(path)
                if pred["record_id"].nunique() != len(eps):
                    print(f"skipping {path.name}: covers {pred['record_id'].nunique()} records")
                    continue
                name = f"BehaviorGPT {variant}" + (" (empty history)" if empty else "")
                predictors[name] = X.bgpt_frame(pred, decisions)
    if args.set == "evaluation":
        choice = p.work / "selection" / "choice.json"
        chosen = json.loads(choice.read_text())["chosen"] if choice.is_file() else None
        if chosen is None or f"BehaviorGPT {chosen}" not in predictors:
            sys.exit(
                "the evaluation records are evaluated only once the chosen variant's "
                "predictions exist (no evaluation result before the labels are fixed)"
            )

    train, dev, news = mind.load_tables(p.work)
    index = B.ExposureIndex([train, dev])
    scores = B.all_baseline_scores(
        decisions, recs, eps, index, B.ContentModel(news), seed=config.SEED
    )
    for name in B.BASELINES:
        decided = B.top_k_decisions(decisions, scores[name].to_numpy(), k_by_record)
        predictors[name] = X.predictor_frame(decisions, scores[name], decided)

    results = {
        "set": args.set,
        "records": len(eps),
        "persons": int(recs["user_id"].nunique()),
        "strata_counts": {k: int(v) for k, v in labels.value_counts().sort_index().items()},
        "predictors": {},
        "comparisons": {},
    }
    per_by_name = {}
    for name, frame in predictors.items():
        agg, per = X.assess(frame, persons, n_boot=args.n_boot, seed=config.SEED)
        agg["E1_by_stratum"] = X.by_stratum(
            per, labels, persons, n_boot=args.n_boot, seed=config.SEED
        )
        sub = X.without_seen_before(frame)
        per_sub = M.ranking_table(sub, "score")
        agg["E1_auc_without_seen_before"] = M.mean_with_ci(
            per_sub["auc"].tolist(),
            persons.loc[per_sub["record_id"]].tolist(),
            n_boot=args.n_boot,
            seed=config.SEED,
        ).as_dict()
        results["predictors"][name] = agg
        per_by_name[name] = per
        print(f"{name:40s} AUC {fmt(agg['E1']['auc'])}", flush=True)

    bgpt = [n for n in per_by_name if n.startswith("BehaviorGPT")]
    pairs = [(a, b) for a in bgpt if "empty" not in a for b in ("random", "popularity")]
    pairs += [(f"BehaviorGPT {v}", f"BehaviorGPT {v} (empty history)") for v in config.VARIANTS]
    pairs += [
        ("BehaviorGPT V2", "BehaviorGPT V1"),
        ("BehaviorGPT V1o", "BehaviorGPT V1"),
        ("BehaviorGPT V2", "BehaviorGPT V1o"),
    ]
    for a, b in pairs:
        if a in per_by_name and b in per_by_name:
            results["comparisons"][f"{a} - {b}"] = X.compare(
                per_by_name[a], per_by_name[b], persons, n_boot=args.n_boot, seed=config.SEED
            )

    out = p.work / "eval" / args.set
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.md").write_text(markdown(results))
    print(markdown(results))
    print(f"-> {(out / 'results.json').relative_to(p.root)}")


if __name__ == "__main__":
    main()
