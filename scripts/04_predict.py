"""Phase 4: BehaviorGPT predicts the attention decisions (the pseudo-labels).

    source .temp/env_setup/devenv.sh
    python -I scripts/04_predict.py estimate --set selection evaluation
    python -I scripts/04_predict.py run --set probe --variants V1 V1o V2
    python -I scripts/04_predict.py run --set selection --approved            # D16
    python -I scripts/04_predict.py select                                    # choose
    python -I scripts/04_predict.py run --set evaluation --variants chosen --approved
    (E3a: the same run with --empty-history added)

Runs on the selection and evaluation records are the full-scale inference and refuse to
start without --approved (the owner's explicit approval). Every call is cached under
.temp/cache/, so a stopped run resumes when started again. Predictions go to
.temp/work/predictions/<set>/<variant>[_empty].parquet with a .json of run facts.
"""

import argparse
import json
import sys

from behaviorgpt import UnboxAIClient

from significance import config, runner
from significance.teacher.api import ResponseCache, Teacher, Throttle


def make_teacher(p, *, rate: float | None, offline: bool = False) -> Teacher:
    state = json.loads((p.work / "catalog_state.json").read_text())
    config.load_env()
    client = None
    if not offline:
        config.require_secret("UNBOXAI_API_KEY")
        client = UnboxAIClient(market=config.MARKET)
    return Teacher(
        client,
        state["catalog_id"],
        cache=ResponseCache(p.cache / "behaviorgpt.sqlite"),
        throttle=Throttle(rate),
        offline=offline,
    )


def chosen_variant(p) -> str | None:
    path = p.work / "selection" / "choice.json"
    return json.loads(path.read_text())["chosen"] if path.is_file() else None


def resolve_variants(p, record_set: str, names) -> list[str]:
    try:
        return runner.variants_for_run(record_set, names, chosen_variant(p))
    except ValueError as exc:
        sys.exit(str(exc))


def cmd_estimate(p, args) -> None:
    teacher = make_teacher(p, rate=None, offline=True)
    report = {}
    for record_set in args.set:
        eps, _, _ = runner.load_set(p.work, record_set)
        if args.variants:
            variants = resolve_variants(p, record_set, args.variants)
        else:
            variants = list(config.DEFAULT_VARIANTS) if record_set != "evaluation" else ["<chosen>"]
        if variants == ["<chosen>"]:
            # one variant runs on the evaluation records; V1/V1o/V2 need the same requests
            report[record_set] = runner.estimate(eps, ["V1"], teacher=teacher)
            report[record_set]["note"] = "requests are the same for V1, V1o and V2"
        else:
            report[record_set] = runner.estimate(
                eps, variants, empty_history=args.empty_history, teacher=teacher
            )
    teacher.cache.close()
    print(json.dumps(report, indent=1))


def cmd_run(p, args) -> None:
    runner.require_approval(args.set, args.approved)
    variants = resolve_variants(p, args.set, args.variants)
    if "V3" in variants and not args.allow_v3:
        sys.exit("V3 failed its Phase 0 go/no-go probe; pass --allow-v3 to run it anyway")
    eps, _, _ = runner.load_set(p.work, args.set)
    if args.limit:
        eps = eps[: args.limit]
    teacher = make_teacher(p, rate=args.rate)
    for variant in variants:
        step = max(1, len(eps) // 10)

        def progress(done, total, variant=variant, step=step):
            if done % step == 0 or done == total:
                s = teacher.stats.as_dict()
                print(
                    f"  {variant}: {done}/{total} records; calls {s['api_calls']}, "
                    f"cache hits {s['cache_hits']}, retries {s['retries']}",
                    flush=True,
                )

        rows, facts = runner.run_variant(
            teacher,
            eps,
            variant,
            empty_history=args.empty_history,
            workers=args.workers,
            progress=progress,
        )
        facts["record_set"] = args.set
        facts["limit"] = args.limit
        path = runner.predictions_path(p.work, args.set, variant, args.empty_history, args.limit)
        runner.save_predictions(rows, facts, path)
        print(json.dumps(facts, indent=1))
        print(f"-> {path.relative_to(p.root)}")
    teacher.cache.close()


def cmd_select(p, args) -> None:
    eps, decisions, recs = runner.load_set(p.work, "selection")
    import pandas as pd

    preds = {}
    for variant in config.VARIANTS:
        path = runner.predictions_path(p.work, "selection", variant, False)
        if path.is_file():
            pred = pd.read_parquet(path)
            if pred["record_id"].nunique() != len(eps):
                sys.exit(
                    f"{variant}: predictions cover {pred['record_id'].nunique()} of "
                    f"{len(eps)} selection records"
                )
            preds[variant] = pred
    if not preds:
        sys.exit("no selection predictions found")
    best, table = runner.choose_variant(preds, decisions, recs)
    out = {"chosen": best, "mean_auc": table, "rule": "best mean per-record AUC (D16)"}
    path = p.work / "selection" / "choice.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="BehaviorGPT predictions (Phase 4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="count the ranking requests a run needs")
    e.add_argument(
        "--set",
        nargs="+",
        default=["selection", "evaluation"],
        choices=["selection", "evaluation", "probe"],
    )
    e.add_argument("--variants", nargs="+")
    e.add_argument("--empty-history", action="store_true")
    r = sub.add_parser("run", help="decode a record set")
    r.add_argument("--set", required=True, choices=["selection", "evaluation", "probe"])
    r.add_argument(
        "--variants",
        nargs="+",
        help="V1 V1o V2 V3 or 'chosen'; the evaluation records run the chosen one only",
    )
    r.add_argument("--empty-history", action="store_true", help="E3a: no previous decisions")
    r.add_argument("--limit", type=int, help="only the first N records")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--rate", type=float, default=15.0, help="max calls per second")
    r.add_argument("--approved", action="store_true", help="owner approved this full run")
    r.add_argument("--allow-v3", action="store_true")
    sub.add_parser("select", help="choose the variant on the selection records")
    args = ap.parse_args()
    p = config.paths()
    {"estimate": cmd_estimate, "run": cmd_run, "select": cmd_select}[args.cmd](p, args)


if __name__ == "__main__":
    main()
