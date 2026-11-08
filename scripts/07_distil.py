"""Phase 7: E4, distilling significance into a supervised student (plans/R1.md 8).

    source .temp/env_setup/devenv.sh
    python -I scripts/07_distil.py                  # the evaluation records
    python -I scripts/07_distil.py --set selection  # functional check

Trains the same logistic-regression student on four label sources and evaluates each on
the REAL decisions of held-out persons (80/20 split by person): BehaviorGPT's
pseudo-labels, the real decisions (upper bound), popularity labels and random labels (both
top-k with the same k rule). Recovery = (AUC_pseudo - 0.5) / (AUC_real - 0.5).

On the evaluation records it runs only once the chosen variant's predictions exist (no
evaluation result before the labels are fixed). On another set the pseudo-labels are used
when that set has predictions of the variant; otherwise only the other three students
run. Output, aggregates only: .temp/work/distil/<set>/results.json and results.md.
"""

import argparse
import json
import sys

import pandas as pd

from significance import baselines as B
from significance import config, runner
from significance import episodes as E
from significance import student as S
from significance.data import mind, records


def fmt(est: dict) -> str:
    lo, hi = est["ci95"]
    return f"{est['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="E4 distillation")
    ap.add_argument(
        "--set",
        default="evaluation",
        choices=["evaluation", "selection", "probe", *records.VIEW_SETS],
    )
    ap.add_argument("--variant", help="pseudo-label variant (default: the chosen one)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    p = config.paths()

    choice = p.work / "selection" / "choice.json"
    variant = args.variant or (
        json.loads(choice.read_text())["chosen"] if choice.is_file() else None
    )
    pred_path = runner.predictions_path(p.work, args.set, variant, False) if variant else None
    have_pseudo = pred_path is not None and pred_path.is_file()
    if args.set == "evaluation" and not have_pseudo:
        sys.exit(
            "the evaluation records are used only once the chosen variant's predictions "
            "exist (no evaluation result before the labels are fixed)"
        )

    eps, dec, recs = runner.load_set(p.work, args.set)
    dec = dec.reset_index(drop=True)
    persons = recs.set_index("record_id")["user_id"]
    train, dev, news = mind.load_tables(p.work)
    index = B.ExposureIndex([train, dev])
    content = B.ContentModel(news)
    k_by_record = {(ep.record_set, ep.record_id): min(E.episode_k(ep), ep.n) for ep in eps}

    labels = {"real": dec["attended"].to_numpy()}
    if have_pseudo:
        pred = pd.read_parquet(pred_path)
        merged = dec.merge(
            pred[["record_set", "record_id", "news_id", "predicted_attention"]],
            on=["record_set", "record_id", "news_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["predicted_attention"].isna().any():
            sys.exit("the predictions do not cover every observation of the set")
        labels["pseudo"] = merged["predicted_attention"].astype(int).to_numpy()
    pop = B.popularity_scores(dec, recs, index)
    labels["popularity"] = B.top_k_decisions(dec, pop, k_by_record)
    labels["random"] = B.top_k_decisions(dec, B.random_scores(dec, config.SEED), k_by_record)

    numeric = S.numeric_features(dec, eps, recs, news, index)
    features = S.Features(news, content)
    train_ids, test_ids = S.split_by_person(recs, 0.2, seed=config.SEED)
    out = S.distil(
        dec,
        numeric,
        features,
        persons,
        labels,
        train_ids,
        test_ids,
        n_boot=args.n_boot,
        seed=config.SEED,
    )
    out["set"] = args.set
    out["pseudo_variant"] = variant if have_pseudo else None

    lines = [
        f"# E4 distillation: {args.set} records",
        "",
        f"Train {out['train_records']} records, test {out['test_records']} records "
        "(split by person); every student is evaluated on the real decisions.",
        "",
        "| Student trained on | AUC (per record) | pooled AUC |",
        "|---|---|---|",
    ]
    for name, r in out["students"].items():
        lines.append(f"| {name} | {fmt(r['auc'])} | {r['pooled_auc']:.3f} |")
    for key in ("recovery", "recovery_popularity", "recovery_random"):
        if key in out:
            lines.append(f"\n{key}: {out[key]['value']:.3f} {out[key]['ci95']}")
    target = p.work / "distil" / args.set
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(out, indent=1))
    (target / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
