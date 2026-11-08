"""Phase 6: write the pseudo-label dataset (plans/R1.md 7.2).

    source .temp/env_setup/devenv.sh
    python -I scripts/06_dataset.py

Reads the chosen variant's predictions on the 5,000 evaluation records and writes
outputs/mind_attention_pseudolabels.parquet with the six columns of section 7.2 only
(record_id, news_id, predicted_attention, attention_rank, confidence, variant). The file
is git-ignored: never commit, push or redistribute it (README "Rights notice").
"""

import json
import sys

import pandas as pd

from significance import config, dataset, runner


def main() -> None:
    p = config.paths()
    choice = p.work / "selection" / "choice.json"
    if not choice.is_file():
        sys.exit("no variant chosen yet (04_predict.py select)")
    variant = json.loads(choice.read_text())["chosen"]
    path = runner.predictions_path(p.work, "evaluation", variant, False)
    if not path.is_file():
        sys.exit(f"no predictions of {variant} on the evaluation records yet")
    predictions = pd.read_parquet(path)
    if predictions["record_id"].nunique() != config.N_EVALUATION_RECORDS:
        sys.exit("the predictions do not cover all evaluation records")
    table = dataset.build_output(predictions, variant)
    out = dataset.write_output(table, p.outputs)
    print(
        f"{table.num_rows} rows, {predictions['record_id'].nunique()} records, variant "
        f"{variant} -> {out.relative_to(p.root)} (git-ignored)"
    )


if __name__ == "__main__":
    main()
