# Data card: MIND attention pseudo-labels (R1)

> **Status:** generated locally on 2026-11-08: 5,000 records, variant V1o. The file is
> git-ignored and has not been distributed. This card holds no data; the full results are in
> [`reports/R1_results.md`](reports/R1_results.md).

## What the labels mean

Each label is **BehaviorGPT's prediction of a human attention decision**: whether a person,
shown a set of news headlines on Microsoft News, chose to open a given headline. The real
decision is recorded in MIND (clicked / not clicked). BehaviorGPT does not define what is
significant. It predicts the person's own decision from:

- the sequence of observations the person encountered before;
- the person's previous decisions.

The predictions are the pseudo-labels. Plan: [`plans/R1.md`](plans/R1.md) §0–§7.

The labels are **not** purchase likelihood, relevance, surprise, or a difference between
history-conditioned and unconditional scores.

## The file

`outputs/mind_attention_pseudolabels.parquet` is git-ignored and is never committed, pushed
or redistributed. It has one row per presented headline of 5,000 decision episodes.

| Column | Type | Origin | Meaning |
|---|---|---|---|
| `record_id` | int64 | created | Number of the decision episode: its position in a seeded random draw, not MIND's impression id. It is not anonymous: with MIND and this code the draw can be redone, and a record's set of news ids identifies its impression. |
| `news_id` | string | original | MIND news id. This is the only original field. |
| `predicted_attention` | int8 | created | 1 if BehaviorGPT predicts the person attended to the headline, else 0. **This is the pseudo-label.** |
| `attention_rank` | int32 | created | Step at which BehaviorGPT chose the headline as the person's next action (1 = first). Unchosen headlines follow, ordered by their last scores. |
| `confidence` | float64 | created | 1 − (`attention_rank` − 1) / *n*. It is based on rank because the API returns ranking scores, not probabilities. |
| `variant` | string | created | The representation that produced the labels (provenance). |

**Not in the file:** person ids, times, histories, titles, categories, or the real decisions.
A schema check (`significance.dataset.validate_output`) enforces this. The real decisions stay
in a local evaluation table under `.temp/work/`, which the deletion term covers.

Records can still be traced back to MIND impressions (see `record_id`). The file is therefore
MIND-derived data and is never distributed.

## Records

- **5,000 evaluation records.** Each is one impression of the MIND-small validation file
  (2019-11-15), drawn uniformly with a fixed seed. 4,833 distinct persons; 37 headlines per
  impression on average.
- **1,000 selection records** from the training file's last day (2019-11-14) choose the
  representation. They never appear in the file.
- **Previous decisions** for each record:
  - MIND's History clicks (weeks 1–4, oldest first);
  - the person's training-file impressions on days strictly before the record's day.

  No impression of the record's own day, and no validation-file impression, enters any
  history. So no record's real decisions reach BehaviorGPT.
- **Composition (Phase 1).** In the evaluation records:
  - 11.9 % of persons have earlier impressions with presented sets;
  - 85.5 % have History clicks only;
  - 2.7 % have no previous decisions at all.

  In the selection records, 82.1 % have earlier impressions.

## How the labels were made

1. **Catalog.** Every MIND-small headline (65,238) was uploaded as a text-only item of a
   private Unbox AI catalog:
   - name = title;
   - categories = "category, subcategory";
   - keywords = the title's entities;
   - no popularity, price or image.

   A `similar_items` spot check found 88 % of neighbours in the same MIND category, against
   14.5 % by chance.
2. **Representation.** Previous decisions are written as BehaviorGPT events, with one session
   per earlier episode, oldest first:
   - **V1:** a View of each attended headline.
   - **V1o:** Views, then an Order of the attended headlines.
   - **V2:** a View of every presented headline, then an Order of the attended ones. A
     headline that was not attended is left out of the Order.
   - **V3:** adds the current screen. It failed its go/no-go probe and is not used.

   The variant with the best mean per-record AUC on the 1,000 selection records is used:
   **V1o** (AUC 0.600, against 0.599 for V1 and 0.569 for V2).
3. **History cap.** BehaviorGPT reads only about the last 256 events of a history (measured
   in Phase 0). Histories are therefore capped at *K* = 256 events, current session included.
   The most recent sessions are kept whole.
4. **Next-action decoding.** BehaviorGPT ranks the presented headlines as the person's next
   action. The top one is attended. It is added to the current session, and the rest are
   ranked again. This repeats *k* times.
5. **How many: *k*.**
   - *k* = max(1, round(*a* × *n*)), with *a* the person's attention rate in their earlier
     impressions and *n* the headlines shown.
   - *k* = 1 without earlier impressions; this applies to 95 % of the evaluation records.
   - The API has no "attend to nothing" action, so *k* cannot come from BehaviorGPT.
6. **Calls.**
   - `complete(history, limit=len(ids), filters={"id": ids}, catalog_id=…, predict="view")`;
   - at most 100 ids per call; chunks share one history and therefore one scale;
   - no user id, so nothing is recorded for any person;
   - every response is cached.

BehaviorGPT itself is unchanged: there is no fine-tuning or training of any kind.

## Evaluation results

These are on the 5,000 evaluation records, against the real decisions, with 95 % bootstrap
intervals over persons. The full tables are in [`reports/R1_results.md`](reports/R1_results.md).

| Predictor | Per-record AUC | F1 | κ |
|---|---|---|---|
| **BehaviorGPT V1o (these labels)** | **0.600** [0.591, 0.609] | 0.130 | 0.098 |
| BehaviorGPT V1o, empty history | 0.523 [0.514, 0.531] | 0.081 | 0.047 |
| Popularity (earlier click rate) | 0.658 [0.650, 0.666] | 0.155 | 0.124 |
| Content similarity | 0.557 [0.548, 0.565] | 0.126 | 0.093 |
| Recency | 0.510 [0.502, 0.517] | 0.066 | 0.032 |
| Random | 0.500 [0.491, 0.508] | 0.080 | 0.046 |

- **Decisions.**
  - precision 0.149, recall 0.115, accuracy 0.937;
  - the *k* rule matches the true number of clicks in 69.7 % of records;
  - told the true count, precision would be 0.158, so the ranking, not the count, limits the
    labels.
- **Contributions (E3).**
  - previous decisions +0.078 AUC (with against without history);
  - writing the ignored headlines (V2) −0.029 against V1 (selection records);
  - the order event +0.001.
- **By stratum.** AUC 0.597 with earlier impressions, 0.603 with History clicks only, 0.527
  without previous decisions.
- **Distillation (E4).** A student trained on these labels reaches per-record AUC 0.580 on
  held-out persons' real decisions, against 0.694 when trained on the real decisions.
  - Recovery 0.41 [0.32, 0.49], below the 50 % target.
  - A student trained on popularity labels recovers 0.81.
- **Reproducibility.** Re-decoding every record from the response cache, with no API calls,
  gives identical labels.

**In short:** the labels carry a real, history-driven attention signal: clearly above chance
and content similarity, but below the popularity baseline. They distil less than half of what
real labels teach.

## Limitations

- **A choice among the presented headlines, not a yes/no.** BehaviorGPT ranks; a single
  headline is never judged on its own (plan L1). Scores are not probabilities (L2).
- **The count *k* comes from the person's past, not from the model** (L3).
  - On the selection records, the rule's *k* equals the true number of clicks in 57 % of
    episodes.
  - The oracle-*k* diagnostic separates count errors from ranking errors.
- **"Not attended" has no event** (L4). It is expressed by leaving a headline out (V1, V1o) or
  out of the Order (V2).
- **What else was on screen is not a model input in V1, V1o and V2** (L5).
- **Only about the last 256 events of a history are read.** Long histories are cut.
- **The first choice continues the latest earlier session.** The SDK cannot open a new session
  without an event in it. BehaviorGPT's first choice in an episode is therefore read as the
  next action after the person's most recent earlier session; for V1o and V2 that session
  ends with an Order. The first choice is the only one for the 95 % of evaluation records
  with *k* = 1.
- **Re-presented headlines stay in the history.** A headline the person was shown (V2) or
  attended (V1, V1o) in an earlier episode stays in its earlier session, even when it is
  presented again now (plan L7). This affects 0.6 % of evaluation and 4.7 % of selection
  observations; results without them are reported separately.
- **Text only.** The model reads the headline text, category and entities. Screen position
  (MIND shuffles it), freshness, time of day and real timing are invisible (L8, L9).
- **Most evaluation records have no earlier impressions.** For them the variants differ
  little, and the observation sequence (V2) cannot contribute.
- **Pretraining domain.** BehaviorGPT is pretrained on retail, engagement and payments. News
  headlines are engagement-like but not products (L12).
- **Repeated calls differ slightly.** Repeated calls with the same input vary by up to about
  0.0003 in score (Phase 0), so near-ties can flip between runs. The response cache makes
  each run reproducible.

## Known biases

- **The population.** MIND-small samples Microsoft News users in the US with at least 5 clicks
  in six weeks (Oct–Nov 2019): English-language, news-reading behaviour, not shopping.
- **No popularity in the catalog.** The catalog holds no popularity signal, so the labels
  carry no direct popularity prior, which matters a lot for news. Popularity is a baseline
  instead.
- **Uneven histories.** Persons with long histories are capped at the most recent 256 events;
  persons without previous decisions are ranked on headline content alone.
- **The model's own taste.** Any taste BehaviorGPT learned in pretraining enters the labels.

## Rights and deletion

- **MIND licence.** MIND belongs to Microsoft and is licensed under the Microsoft Research
  License Terms: non-commercial research only, **no distribution**, and results may be
  published only without a material portion of the dataset.
- **The labels.** The labels are BehaviorGPT outputs, subject to Unbox AI's terms. The MIT
  licence of this repository covers code only.
- **Never distributed.** The label file is never committed, pushed or redistributed.
- **Deletion.** See [`README.md`](README.md):
  - `python -I scripts/delete_data.py --confirm` deletes `.temp/data/`, `.temp/work/`,
    `.temp/cache/`, `outputs/` and the project's own Unbox AI catalog;
  - on a rights holder's request, if the terms disallow this use, and at the end of the
    project.

## Regenerating the dataset locally

You need MIND access, meaning the MIND licence accepted on Hugging Face and `HF_TOKEN` in the
git-ignored `.env`, and an Unbox AI key (`UNBOXAI_API_KEY` in `.env`).

```sh
source .temp/env_setup/devenv.sh                 # project venv; caches stay in .temp/
python -m pip install --no-deps -e .             # once
python -I scripts/01_prepare.py                  # download, check, describe, draw records
python -I scripts/02_build.py                    # catalog file and decision episodes
python -I scripts/03_embed.py slice && python -I scripts/03_embed.py full
python -I scripts/03_embed.py check
python -I scripts/04_predict.py estimate         # exact number of API requests
python -I scripts/04_predict.py run --set selection --approved
python -I scripts/04_predict.py select
python -I scripts/04_predict.py run --set evaluation --variants chosen --approved
python -I scripts/04_predict.py run --set evaluation --variants chosen --empty-history --approved
python -I scripts/05_evaluate.py --set evaluation
python -I scripts/06_dataset.py                  # writes outputs/mind_attention_pseudolabels.parquet
python -I scripts/07_distil.py
```

Responses are cached under `.temp/cache/`, so a rerun reproduces the same labels without new
calls.
