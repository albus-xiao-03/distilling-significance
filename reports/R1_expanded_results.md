# R1, expanded: a 25,000-impression attention pseudo-label dataset

2026-11-08. This extends [`R1_results.md`](R1_results.md), the first 5,000-record dataset,
on the owner's instruction. The dataset grows to **25,000 MIND-small impressions**, split
between the training and validation files in proportion to their sizes. Everything already
processed, cached or labelled is reused. The representation (V1o), the inference method, the
chronological rules and the evaluation procedures are unchanged.

**Aggregate numbers only:** no headlines, ids or per-impression rows. All intervals are 95 %
bootstrap intervals over persons.

## 1. Scope

| MIND-small file | Impressions in the file | Share | Dataset records |
|---|---|---|---|
| Training (2019-11-09 to 11-14) | 156,965 | 68.2 % | **17,053** |
| Validation (2019-11-15) | 73,152 | 31.8 % | **7,947** |
| **Total** | 230,117 | | **25,000** |

The counts use largest-remainder rounding of 25,000 × the files' shares.

| Records | Where from | How drawn | Labels |
|---|---|---|---|
| 5,000 evaluation records | validation file | uniform (R1) | **reused** |
| 2,947 more | validation file | uniform among the rest of the file | new run |
| 1,000 selection records | training file, 11-14 | uniform over that day (R1) | **reused** (their V1o run) |
| 20 probe records | training file, 11-14 | uniform over that day (R1) | **reused** |
| 16,033 more | training file, all six days | a proportional quota per day | new run |

**Training records per day.** The quota is proportional to each day's impressions, and the
reused last-day records count towards 11-14:

| Day | 11-09 | 11-10 | 11-11 | 11-12 | 11-13 | 11-14 |
|---|---|---|---|---|---|---|
| Records | 1,474 | 1,635 | 3,563 | 3,656 | 3,436 | 3,289 |

**Unchanged rules.**

- **Previous decisions:** MIND History clicks, then the person's training-file impressions on
  days strictly before the record's day. A training-file record on day *d* therefore sees only
  days before *d*. No record's own clicks ever enter its own prediction.
- **Method:** representation V1o, *k* rule, history cap 256 events, next-action decoding.

## 2. Runs and API usage

| Run | Records | Ranking requests | New API calls | Cache hits | Retries / failures |
|---|---|---|---|---|---|
| New validation records (V1o) | 2,947 | 3,933 | 3,906 | 27 | 0 / 0 |
| New validation records, empty history (E3a) | 2,947 | 3,933 | 2,955 | 978 | 0 / 0 |
| New training records (V1o) | 16,033 | 33,661 | 33,658 | 3 | 0 / 0 |
| **Expansion total** | | **41,527** | **40,519** | **1,008** | **0 / 0** |
| Reused labels: evaluation, selection, probe records | 6,020 | — | **0** | — | — |

- **Duration.** About 34 minutes at 20 calls/s. The requests equal the estimate made before
  the run.
- **No id was ever missing** from a response.
- **Probe responses.** The probe records' responses dated from the first API checks. They were
  re-keyed in the cache instead of being requested again.

**All label calls** behind the 25,000 records, R1 and the expansion together: 46,537 API
calls.

**Reproducibility.** Every one of the 25,000 records was decoded again from the response
cache alone (46,575 cache hits, **0 API calls**). Its labels were identical, and identical to
the dataset files row for row (925,760 rows).

## 3. The dataset

| File (`dataset/`, git-ignored) | Records | Rows | Labelled attended |
|---|---|---|---|
| `train.parquet` | 17,053 | 631,219 | 31,320 (4.96 %) |
| `validation.parquet` | 7,947 | 294,541 | 9,382 (3.19 %) |

- **Columns.** Each file holds the six columns of plan §7.2, and the per-record label checks
  pass. `metadata.json` holds the scope, split rule, schema, record-id ranges by origin,
  method, API usage and sha256 checksums.
- **Record ids.** 0–7,946 are the validation file; the first 5,000 keep their ids from the
  first dataset. 7,947–24,999 are the training file. The selection records are 7,947–8,946.
- **Composition.**

  | Records | With earlier impressions | History clicks only | No previous decisions | Mean *k* |
  |---|---|---|---|---|
  | Validation file | 12.4 % | 84.9 % | 2.7 % | 1.18 |
  | Training file | 55.5 % | 43.7 % | 0.8 % | 1.84 |

## 4. Evaluation design

The procedures of R1 are unchanged: E1/E2 against the real decisions, the four baselines, AUC
by stratum and without re-presented headlines, E3a and E4. They run on views of the dataset:

| View | Records | Used for |
|---|---|---|
| `dataset_eval` | 24,000 | The main evaluation and E4. The 1,000 selection records are left out, because their real decisions chose V1o. |
| `dataset_validation` | 7,947 | The validation-file part, and E3a (with against without history). E3a is defined on validation-file records, as in R1. |
| `dataset_training` | 16,053 | The training-file part, without the selection records. |

## 5. Results

### E1–E2 on the 24,000 evaluation records

| Predictor | AUC | MRR | nDCG@10 | F1 | κ |
|---|---|---|---|---|---|
| **BehaviorGPT V1o** | **0.582** [0.578, 0.585] | 0.277 | 0.353 | 0.118 [0.114, 0.121] | 0.079 |
| Popularity | 0.662 [0.659, 0.666] | 0.324 | 0.409 | 0.164 | 0.128 |
| Content similarity | 0.573 [0.569, 0.576] | 0.286 | 0.361 | 0.136 | 0.098 |
| Recency | 0.509 [0.505, 0.513] | 0.232 | 0.302 | 0.088 | 0.048 |
| Random | 0.502 [0.498, 0.506] | 0.229 | 0.294 | 0.084 | 0.044 |

- **Paired differences.** BehaviorGPT − random = +0.080 [0.074, 0.085]. BehaviorGPT −
  popularity = −0.081 [−0.086, −0.076].
- **Decisions.**
  - precision 0.114, recall 0.122;
  - the *k* rule gives 1.61 attended per record against 1.50 real clicks, and matches the true
    count in 65.4 % of records;
  - oracle-*k* precision is 0.142.
- **Re-presented headlines.** Leaving out headlines seen in an earlier episode (3.1 % of
  observations, in 20.9 % of records) leaves AUC unchanged at 0.582.

### By MIND file

| Records | BehaviorGPT V1o | Popularity | Content | Recency | Random |
|---|---|---|---|---|---|
| Validation file (7,947) | **0.599** [0.592, 0.606] | 0.655 | 0.554 | 0.508 | 0.497 |
| Training file (16,053) | **0.573** [0.568, 0.578] | 0.666 | 0.584 | 0.510 | 0.504 |

On the validation file, BehaviorGPT replicates R1 (0.600 on 5,000 records → 0.599 on 7,947).
On the training file it is lower, and below content similarity.

### By stratum (24,000 records)

| Stratum | Records | BehaviorGPT V1o | Popularity | Content |
|---|---|---|---|---|
| Earlier impressions | 9,620 | 0.583 | 0.656 | 0.590 |
| History clicks only | 14,028 | 0.583 | 0.668 | 0.562 |
| No previous decisions | 352 | 0.510 | 0.625 | 0.502 |

### By day: training-file records, a diagnostic

| Day | 11-09 | 11-10 | 11-11 | 11-12 | 11-13 | 11-14 | 11-15 (validation) |
|---|---|---|---|---|---|---|---|
| BehaviorGPT V1o | 0.530 | 0.530 | 0.595 | 0.553 | 0.585 | 0.611 | 0.599 |
| Content similarity | 0.585 | 0.572 | 0.596 | 0.571 | 0.591 | 0.583 | 0.554 |

- **The day explains the drop.** BehaviorGPT's accuracy swings from day to day, much more than
  content similarity's. The lower training-file average comes from 11-09, 11-10 and 11-12.
- **The decoding does not.** Records needing several decoding steps (*k* > 1: 0.582) are not
  worse than single-step ones (0.570). The cause is not yet explained.

### E3a: what previous decisions contribute (7,947 validation-file records)

| Predictor | AUC |
|---|---|
| BehaviorGPT V1o with the person's history | 0.599 [0.592, 0.606] |
| The same with an empty history | 0.521 [0.514, 0.527] |
| **Difference** | **+0.079** [0.070, 0.087] (R1: +0.078) |

### E4: distillation (24,000 records)

The student was trained on 19,215 records and evaluated on the real decisions of 4,785
records of held-out persons.

| Student trained on | Per-record AUC | Pooled AUC |
|---|---|---|
| Real decisions (upper bound) | 0.691 [0.683, 0.699] | 0.765 |
| **BehaviorGPT pseudo-labels** | **0.561** [0.553, 0.570] | 0.653 |
| Popularity labels | 0.660 [0.651, 0.668] | 0.712 |
| Random labels | 0.517 [0.508, 0.526] | 0.673 |

**Recovery** = (AUC_pseudo − 0.5) / (AUC_real − 0.5) = **0.32** [0.28, 0.36]. R1 had 0.41
on 5,000 validation records. Popularity labels recover 0.84.

## 6. Success targets (plan §8), on the expanded dataset

| Target | Result |
|---|---|
| 1. Every record predicted end to end, cached, reproducible | **Met.** 25,000 records, 0 failures; the offline re-decode is identical. |
| 2. Beats random, CI excluding 0.5; compared with popularity | **Met** for random (0.582 [0.578, 0.585]). Popularity is better by 0.081. |
| 3. E3 reported | **Met.** History +0.079 (validation file); R1's representation comparisons stand. |
| 4. Recovery ≥ 50 % | **Not met.** 0.32 [0.28, 0.36]. |

## 7. Reading the results

- **The validation-file result is stable.** The R1 estimate replicates on 2,947 new
  validation records: AUC 0.599 against 0.600, and history +0.079 against +0.078.
- **The training file is harder for BehaviorGPT, and the day matters.**
  - On the six training days, its accuracy ranges from 0.53 to 0.61.
  - Content similarity and popularity barely move.
  - So the 0.600 of R1 was not a property of all MIND-small days.
- **Popularity remains far ahead everywhere** (0.66), and content similarity is now close to
  BehaviorGPT overall.
- **The larger dataset distils less.** Recovery falls to 0.32, as the weaker training-file
  labels enter the student's training data.

## 8. Limitations and next steps

The R1 limitations hold (plan §4; data card). New with the expansion:

- **Unexplained day effect.** The day-to-day variation is observed but not explained. A next
  step is to test it against the day's news mix and popularity concentration.
- **Uneven earlier impressions.** Training-file records have earlier impressions far more
  often than validation-file ones (55.5 % against 12.4 %), and fewer for the earliest days.
- **E3a scope.** E3a stays on the validation-file records. Extending it to the 16,033 new
  training-file records would take about 33,700 further requests.

Next steps from R1 stand: give BehaviorGPT a leakage-safe popularity signal; combine its
personal signal with popularity; try explicit skip/attend items; repeat on other datasets.
