# R1 results: BehaviorGPT's predicted attention decisions on MIND-small

2026-11-08. Plan: [`plans/R1.md`](../plans/R1.md). Feasibility facts:
[`feasibility.md`](feasibility.md). **Aggregate numbers only:** no headlines, ids or
per-impression rows. All intervals are 95 % bootstrap intervals over persons (1,000
resamples). Each of a person's records moves together.

## 1. What was run

| Run | Records | Ranking requests | New API calls | Cache hits | Retries / failures | Wall time |
|---|---|---|---|---|---|---|
| Selection: V1, V1o, V2 | 1,000 × 3 | 7,326 | 7,126 | 200 | 0 / 0 | 7 min 56 s |
| Evaluation: V1o (chosen) | 5,000 | 6,511 | 6,506 | 5 | 0 / 0 | 7 min 14 s |
| Evaluation, empty history (E3a) | 5,000 | 6,511 | 5,140 | 1,371 | 0 / 0 | 5 min 44 s |
| **Total** | | **20,348** | **18,772** | **1,576** | **0 / 0** | **≈ 21 min** |

- **Requests.** The counts equal the planned estimate exactly. No id was ever missing from a
  response.
- **Cache hits in the empty-history run.** Without a history, records shown the same headline
  set send the same request: 30 % of evaluation records share their exact set with another
  record, mostly small lineups of 2–11 headlines.
- **Reproducibility.** All 5,000 evaluation records were decoded again from the response
  cache alone, with no API client. Ranks and labels were identical, with 0 calls.
- **The dataset.** `outputs/mind_attention_pseudolabels.parquet` is git-ignored and not
  distributed:
  - 185,373 rows from 5,000 records;
  - the six planned columns only;
  - 5,837 pseudo-label positives (3.15 %; mean *k* = 1.17);
  - the schema and per-record label checks pass.

## 2. Choosing the representation (D16, 1,000 selection records)

| Variant | Mean per-record AUC | Without re-presented headlines |
|---|---|---|
| V1: views of attended headlines | 0.599 [0.582, 0.616] | 0.600 |
| **V1o: views, then an order of attended headlines** | **0.600 [0.583, 0.616]** | **0.601** |
| V2: views of everything shown, then an order of attended ones | 0.569 [0.552, 0.587] | 0.572 |

- **Choice.** V1o is chosen by the planned rule (best mean AUC). Both criteria agree.
- **V3** had been dropped at its Phase 0 go/no-go probe.

## 3. E1–E2: predictions against the real decisions (5,000 evaluation records)

| Predictor | AUC | MRR | nDCG@5 | nDCG@10 | F1 | κ | Oracle-*k* precision |
|---|---|---|---|---|---|---|---|
| **BehaviorGPT V1o** | **0.600** [0.591, 0.609] | 0.289 | 0.312 | 0.372 | 0.130 [0.122, 0.138] | 0.098 | 0.158 |
| BehaviorGPT V1o, empty history | 0.523 [0.514, 0.531] | 0.223 | 0.235 | 0.294 | 0.081 [0.075, 0.088] | 0.047 | 0.097 |
| Popularity: earlier click rate | 0.658 [0.650, 0.666] | 0.317 | 0.348 | 0.408 | 0.155 [0.148, 0.164] | 0.124 | 0.191 |
| Content similarity: TF-IDF | 0.557 [0.548, 0.565] | 0.269 | 0.282 | 0.345 | 0.126 [0.118, 0.134] | 0.093 | 0.146 |
| Recency | 0.510 [0.502, 0.517] | 0.216 | 0.230 | 0.292 | 0.066 [0.060, 0.072] | 0.032 | 0.086 |
| Random | 0.500 [0.491, 0.508] | 0.221 | 0.227 | 0.290 | 0.080 [0.073, 0.086] | 0.046 | 0.096 |

**Decisions (E2).** For BehaviorGPT V1o:

- precision 0.149, recall 0.115, accuracy 0.937, κ 0.098.
- **The count *k*.** The *k* rule gives on average 1.17 attended headlines per record,
  against 1.52 real clicks. It matches the true count in 69.7 % of records; mean absolute
  error 0.61.
- **Count against ranking.** Told the true count, BehaviorGPT would reach precision 0.158
  (oracle *k*), against 0.149 with the *k* rule. The ranking, not the count, limits the
  decisions.

**Re-presented headlines (L7).** Leaving out headlines seen in an earlier episode (0.6 % of
observations) changes BehaviorGPT's AUC by less than 0.001 (0.600).

## 4. E3: what each input contributes

| Comparison (paired AUC difference) | Records | Difference |
|---|---|---|
| (a) Previous decisions: V1o − V1o with empty history | evaluation | **+0.078** [0.067, 0.088] |
| (b) Observation sequence: V2 − V1 | selection | **−0.029** [−0.047, −0.013] |
| (c) Decision event: V1o − V1 | selection | +0.001 [−0.012, 0.014] |
| BehaviorGPT V1o − random | evaluation | +0.101 [0.089, 0.112] |
| BehaviorGPT V1o − popularity | evaluation | **−0.058** [−0.068, −0.047] |

**AUC by stratum** (evaluation records):

| Stratum | Records | BehaviorGPT V1o | Empty history | Popularity |
|---|---|---|---|---|
| Earlier impressions | 594 | 0.597 [0.574, 0.620] | 0.522 | 0.665 |
| History clicks only | 4,273 | 0.603 [0.594, 0.613] | 0.523 | 0.659 |
| No previous decisions | 133 | 0.527 [0.477, 0.578] | 0.527 | 0.610 |

## 5. E4: distilling into a student

The student is a logistic regression on non-BehaviorGPT features. It was trained on 4,001
records and evaluated on the real decisions of 999 records of held-out persons.

| Student trained on | Per-record AUC | Pooled AUC |
|---|---|---|
| Real decisions (upper bound) | 0.694 [0.676, 0.713] | 0.775 |
| **BehaviorGPT pseudo-labels** | **0.580** [0.562, 0.597] | 0.677 |
| Popularity labels (top-*k*) | 0.658 [0.641, 0.676] | 0.741 |
| Random labels (top-*k*) | 0.492 [0.475, 0.509] | 0.684 |

- **Recovery** = (AUC_pseudo − 0.5) / (AUC_real − 0.5) = **0.41** [0.32, 0.49]. Popularity
  labels recover 0.81.
- **Why per-record AUC.** Pooled AUC flatters every student, the random-label one included,
  because the features carry each impression's size, and with it its click rate. Per-record
  AUC is the measure.

## 6. Success targets (plan §8)

| Target | Result |
|---|---|
| 1. 5,000 records predicted end to end, cached, reproducible | **Met.** 0 failures; the offline re-decode is identical. |
| 2. The chosen variant beats random, CI excluding 0.5; compared with popularity | **Met** for random (AUC 0.600 [0.591, 0.609]). Popularity is better by 0.058; beating it was a stretch goal. |
| 3. E3 reported in whichever direction | **Met.** History +0.078; observation sequence −0.029; decision event +0.001. |
| 4. Recovery ≥ 50 % | **Not met.** 0.41 [0.32, 0.49]. |

## 7. Reading the results

- **BehaviorGPT does predict attention from previous decisions, with no training.**
  - With the person's earlier clicks, its ranking of the headlines shown is clearly better
    than chance and better than content similarity: AUC 0.600 against 0.557.
  - Without them it is close to chance (0.523). Almost all of the signal comes from previous
    decisions (E3a).
- **It does not beat popularity.** News attention is dominated by what is popular right now.
  - The catalog deliberately holds no popularity, and BehaviorGPT sees no other readers.
  - The earlier-click-rate baseline therefore wins on every stratum.
- **The observation sequence hurts as written.**
  - V2 writes the headlines a person saw and ignored as views, and the API reads any view as
    interest (plan L4: there is no "not attended" event). This dilutes the person's real
    interests.
  - The order event (V1o) neither helps nor hurts.
  - So "attention as the next view of what was attended" is the representation that works.
- **The pseudo-labels carry less than half of the real labels' learnable signal** (recovery
  0.41). They are better than random labels but worse than popularity labels.
  - As a stand-alone teacher for this task, BehaviorGPT is weaker than a popularity count.
  - Its signal is largely personal (history-based), so it may complement popularity rather
    than replace it.

## 8. Limitations

These are the limitations of plan §4 and the data card, in short:

- **Rankings, not yes/no.** BehaviorGPT ranks; it never judges a single headline (L1, L2).
- ***k* comes from the person's past.** 95 % of evaluation records have *k* = 1 (L3).
- **There is no "not attended" event** (L4).
- **About 256 history events are read.**
- **Text-only items:** no position, freshness or time of day (L8, L9).
- **The first decoding step continues the person's latest earlier session.**
- **Few earlier impressions.** Only 11.9 % of evaluation records have earlier impressions.
- **Population.** One news population: MIND-small, US, Oct–Nov 2019.

## 9. Next steps (not in R1)

- **Give BehaviorGPT popularity.** Upload the catalog with `sales_since`/`frequency` computed
  strictly before the evaluation day, and compare.
- **Combine teachers.** Use BehaviorGPT ranks alongside popularity, as student features or a
  combined label, to test whether its personal signal adds to popularity.
- **Explicit decision items.** Test explicit "skip"/"attend" decision items (plan §12), which
  would give the model a way to express "not attended" and the count *k*.
- **Other datasets.** Repeat on a second news dataset (EB-NeRD) and, once one is verified, on
  a shopping dataset with logged impressions.
