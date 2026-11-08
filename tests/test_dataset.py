"""Output dataset: exact schema (no person, time, text or real decision) and label checks."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from significance import config, dataset

FORBIDDEN = {
    "user_id",
    "time",
    "date",
    "history",
    "title",
    "category",
    "subcategory",
    "attended",
    "impression_id",
    "score",
    "k",
    "record_set",
    "empty_history",
}


def _predictions(variant="V1", record_set="evaluation", empty=False):
    rows = []
    for rid, n, k in ((0, 4, 1), (1, 3, 2)):
        for rank in range(1, n + 1):
            rows.append(
                {
                    "record_set": record_set,
                    "record_id": rid,
                    "news_id": f"N{rid}{rank}",
                    "attention_rank": rank,
                    "predicted_attention": int(rank <= k),
                    "confidence": 1 - (rank - 1) / n,
                    "score": 0.1 * rank,
                    "k": k,
                    "variant": variant,
                    "empty_history": empty,
                    "user_id": "U1",
                    "attended": 1,  # must never reach the output
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=0)


def test_output_has_exactly_the_six_columns_and_no_forbidden_field(tmp_path):
    table = dataset.build_output(_predictions(), "V1")
    assert tuple(table.column_names) == config.OUTPUT_COLUMNS
    assert not FORBIDDEN & set(table.column_names)
    path = dataset.write_output(table, tmp_path)
    assert path.name == "mind_attention_pseudolabels.parquet"
    back = pq.read_table(path)
    assert back.schema.remove_metadata() == dataset.SCHEMA
    df = back.to_pandas()
    assert df["record_id"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert df["attention_rank"].tolist() == [1, 2, 3, 4, 1, 2, 3]


@pytest.mark.parametrize(
    ("kwargs", "variant", "match"),
    [
        ({"record_set": "selection"}, "V1", "evaluation records"),
        ({"empty": True}, "V1", "empty-history"),
        ({}, "V2", "chosen"),
        ({}, "V9", "unknown"),
    ],
)
def test_wrong_inputs_are_refused(kwargs, variant, match):
    with pytest.raises(ValueError, match=match):
        dataset.build_output(_predictions(**kwargs), variant)


def test_validate_output_catches_inconsistent_labels_and_extra_columns():
    good = dataset.build_output(_predictions(), "V1").to_pandas()
    bad_rank = good.copy()
    bad_rank.loc[0, "attention_rank"] = 9
    bad_top = good.copy()
    bad_top.loc[0, "predicted_attention"] = 0
    bad_top.loc[1, "predicted_attention"] = 1
    bad_conf = good.copy()
    bad_conf.loc[0, "confidence"] = 0.5
    for frame, match in ((bad_rank, "ranks"), (bad_top, "top"), (bad_conf, "confidence")):
        with pytest.raises(ValueError, match=match):
            dataset.validate_output(
                pa.Table.from_pandas(frame, schema=dataset.SCHEMA, preserve_index=False)
            )
    extra = pa.Table.from_pandas(good.assign(user_id="U1"), preserve_index=False)
    with pytest.raises(ValueError, match="schema"):
        dataset.validate_output(extra)
