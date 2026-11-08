# Data card: MIND attention pseudo-labels

> **Status:** generated locally on 2026-11-08. It holds **25,000 impressions**: 17,053 from
> MIND-small's training file and 7,947 from its validation file, labelled with variant V1o.
> The dataset folder is git-ignored and has not been distributed. This card holds no data.
> The full results are in [`reports/R1_expanded_results.md`](reports/R1_expanded_results.md);
> the first, 5,000-record version is in [`reports/R1_results.md`](reports/R1_results.md).

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

## The files

The dataset lives in `dataset/`, directly under the project. The folder is git-ignored and is
never committed, pushed or redistributed.

| File | Records | Rows | Labelled attended |
|---|---|---|---|
| `train.parquet` | 17,053 impressions of MIND-small's training file | 631,219 | 31,320 (4.96 %) |
| `validation.parquet` | 7,947 impressions of MIND-small's validation file | 294,541 | 9,382 (3.19 %) |
| `metadata.json` | scope, split rule, schema, record-id ranges by origin, method, API usage, sha256 checksums | | |

Each parquet file has one row per presented headline and exactly these columns:

| Column | Type | Origin | Meaning |
|---|---|---|---|
| `record_id` | int64 | created | Number of the decision episode (one impression), unique across both files: 0–7,946 validation file, 7,947–24,999 training file. It is not MIND's impression id, but it is not anonymous: with MIND and this code the draw can be redone, and a record's set of news ids identifies its impression. |
| `news_id` | string | original | MIND news id. This is the only original field. |
| `predicted_attention` | int8 | created | 1 if BehaviorGPT predicts the person attended to the headline, else 0. **This is the pseudo-label.** |
| `attention_rank` | int32 | created | Step at which BehaviorGPT chose the headline as the person's next action (1 = first). Unchosen headlines follow, ordered by their last scores. |
| `confidence` | float64 | created | 1 − (`attention_rank` − 1) / *n*. It is based on rank because the API returns ranking scores, not probabilities. |
| `variant` | string | created | The representation that produced the labels (`V1o`). |

**Not in the files:** person ids, times, histories, titles, categories, or the real decisions.
A schema check (`significance.dataset.validate_output`) enforces this for each file. The real
decisions stay in local evaluation tables under `.temp/work/`, which the deletion term covers.

Records can still be traced back to MIND impressions (see `record_id`). The folder is
therefore MIND-derived data and is never distributed.

## Records

- **25,000 impressions in MIND-small's proportion.** The training file holds 156,965
  impressions and the validation file 73,152, so 25,000 splits as **17,053 : 7,947**
  (largest-remainder rounding).
- **Validation-file records** (2019-11-15) are a uniform sample of the file:
  - the 5,000 evaluation records of the first dataset (ids unchanged);
  - 2,947 more, drawn uniformly from the rest.
- **Training-file records** (2019-11-09 to 11-14) have a quota per day in proportion to that
  day's impressions:

  | Day | 11-09 | 11-10 | 11-11 | 11-12 | 11-13 | 11-14 |
  |---|---|---|---|---|---|---|
  | Records | 1,474 | 1,635 | 3,563 | 3,656 | 3,436 | 3,289 |

  The 11-14 records include the 1,000 selection records and 20 probe records of the first
  run. Their labels are reused.
- **Selection records.** The 1,000 selection records (record ids 7,947–8,946) are in the
  files, but their real decisions chose the representation. They are therefore left out of
  every evaluation below.
- **Previous decisions** for each record:
  - MIND's History clicks (weeks 1–4, oldest first);
  - the person's training-file impressions on days strictly before the record's day.

  A record never sees its own day or anything later, and its own clicks never enter its
  prediction.
- **Composition.**

  | Records | With earlier impressions | History clicks only | No previous decisions | Mean *k* |
  |---|---|---|---|---|
  | Validation file | 12.4 % | 84.9 % | 2.7 % | 1.18 |
  | Training file | 55.5 % | 43.7 % | 0.8 % | 1.84 |

## How the labels were made

This is unchanged from the first dataset.

1. **Catalog.** Every MIND-small headline (65,238) was uploaded as a text-only item of a
   private Unbox AI catalog:
   - name = title;
   - categories = "category, subcategory";
   - keywords = the title's entities;
   - no popularity, price or image.

   A `similar_items` spot check found 88 % of neighbours in the same MIND category, against
   14.5 % by chance.
2. **Representation.** Previous decisions are written as BehaviorGPT events, one session per
   earlier episode, oldest first. The variant with the best mean per-record AUC on the 1,000
   selection records is used: **V1o**, views of the attended headlines then an order of them
   (AUC 0.600, against 0.599 for V1 and 0.569 for V2). V3 failed its go/no-go probe.
3. **History cap.** BehaviorGPT reads only about the last 256 events of a history (measured
   in Phase 0). Histories are therefore capped at *K* = 256 events, current session included.
   The most recent sessions are kept whole.
4. **Next-action decoding.** BehaviorGPT ranks the presented headlines as the person's next
   action. The top one is attended. It is added to the current session, and the rest are
   ranked again. This repeats *k* times.
5. **How many: *k*.**
   - *k* = max(1, round(*a* × *n*)), with *a* the person's attention rate in their earlier
     impressions and *n* the headlines shown.
   - *k* = 1 without earlier impressions.
   - The API has no "attend to nothing" action, so *k* cannot come from BehaviorGPT.
6. **Calls.**
   - `complete(history, limit=len(ids), filters={"id": ids}, catalog_id=…, predict="view")`;
   - at most 100 ids per call; chunks share one history and therefore one scale;
   - no user id, so nothing is recorded for any person;
   - every response is cached.

   The 25,000 records took **46,537 label calls** in total, 40,519 of them in the expansion,
   and 0 failures. All 25,000 records decode again from the cache alone to identical labels,
   with no call.

BehaviorGPT itself is unchanged: there is no fine-tuning or training of any kind.

## Evaluation results

The 24,000 records without the selection records, against the real decisions, with 95 %
bootstrap intervals over persons.

| Predictor | Per-record AUC | F1 | κ |
|---|---|---|---|
| **BehaviorGPT V1o (these labels)** | **0.582** [0.578, 0.585] | 0.118 | 0.079 |
| Popularity (earlier click rate) | 0.662 [0.659, 0.666] | 0.164 | 0.128 |
| Content similarity | 0.573 [0.569, 0.576] | 0.136 | 0.098 |
| Recency | 0.509 [0.505, 0.513] | 0.088 | 0.048 |
| Random | 0.502 [0.498, 0.506] | 0.084 | 0.044 |

**By file.**

| Records | BehaviorGPT V1o | Popularity | Content |
|---|---|---|---|
| Validation file (7,947) | 0.599 [0.592, 0.606] | 0.655 | 0.554 |
| Training file (16,053) | 0.573 [0.568, 0.578] | 0.666 | 0.584 |

- **Decisions.**
  - precision 0.114, recall 0.122, accuracy 0.926;
  - the *k* rule matches the true number of clicks in 65.4 % of records;
  - told the true count, precision would be 0.142.
- **Previous decisions (E3a, validation file).** +0.079 AUC with against without history,
  replicating the first dataset's +0.078.
- **Day effect.** On the training file, BehaviorGPT's AUC varies by day from 0.53 (11-09,
  11-10) to 0.61 (11-14). Content similarity stays between 0.57 and 0.60.
- **Distillation (E4).** A student trained on these labels reaches per-record AUC 0.561 on
  held-out persons' real decisions, against 0.691 when trained on the real decisions.
  - Recovery 0.32 [0.28, 0.36], below the 50 % target.
  - A student trained on popularity labels recovers 0.84.

**In short:** the labels carry a real, history-driven attention signal: clearly above chance,
and stable on the validation file (0.599, as in the first dataset). They are weaker and
day-dependent on the training file, below popularity everywhere, and they distil about a
third of what real labels teach.

## Limitations

- **A choice among the presented headlines, not a yes/no.** BehaviorGPT ranks; a single
  headline is never judged on its own (plan L1). Scores are not probabilities (L2).
- **The count *k* comes from the person's past, not from the model** (L3). The oracle-*k*
  diagnostic separates count errors from ranking errors.
- **"Not attended" has no event** (L4). It is expressed by leaving a headline out (V1, V1o) or
  out of the Order (V2).
- **What else was on screen is not a model input in V1, V1o and V2** (L5).
- **Only about the last 256 events of a history are read.** Long histories are cut.
- **The first choice continues the latest earlier session.** The SDK cannot open a new session
  without an event in it. BehaviorGPT's first choice in an episode is therefore read as the
  next action after the person's most recent earlier session; for V1o that session ends with
  an Order.
- **Re-presented headlines stay in the history.** A headline the person attended in an
  earlier episode stays in its earlier session, even when it is presented again now (plan
  L7). Results without such headlines are unchanged.
- **Text only.** The model reads the headline text, category and entities. Screen position
  (MIND shuffles it), freshness, time of day and real timing are invisible (L8, L9).
- **Unexplained day effect.** Accuracy varies strongly by day on the training file; the cause
  is not yet known.
- **Uneven earlier impressions.** Training-file records have earlier impressions far more
  often than validation-file ones, and records of the earliest days have fewer.
- **Pretraining domain.** BehaviorGPT is pretrained on retail, engagement and payments. News
  headlines are engagement-like but not products (L12).
- **Repeated calls differ slightly.** Repeated calls with the same input vary by up to about
  0.0003 in score (Phase 0), so near-ties can flip between runs. The response cache makes the
  dataset reproducible.

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
- **Never distributed.** The dataset folder is never committed, pushed or redistributed.
- **Deletion.** See [`README.md`](README.md):
  - `python -I scripts/delete_data.py --confirm` deletes `.temp/data/`, `.temp/work/`,
    `.temp/cache/`, `dataset/`, `outputs/` and the project's own Unbox AI catalog;
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
python -I scripts/04_predict.py run --set evaluation --approved
python -I scripts/04_predict.py run --set evaluation --empty-history --approved
python -I scripts/08_expand.py records           # the extra records (25,000 in total)
python -I scripts/04_predict.py run --set evaluation_extra --approved
python -I scripts/04_predict.py run --set evaluation_extra --empty-history --approved
python -I scripts/04_predict.py run --set training_extra --approved
python -I scripts/08_expand.py views             # assemble the dataset views (no API call)
python -I scripts/06_dataset.py                  # writes dataset/
python -I scripts/05_evaluate.py --set dataset_eval
python -I scripts/05_evaluate.py --set dataset_validation   # with E3a
python -I scripts/05_evaluate.py --set dataset_training
python -I scripts/07_distil.py --set dataset_eval
```

Responses are cached under `.temp/cache/`, so a rerun reproduces the same labels without new
calls.
