# R1 feasibility note: API probes, data facts and the request budget

2026-11-08. Phases 0–3 of [`plans/R1.md`](../plans/R1.md), plus small API checks on the MIND
catalog. **Aggregate numbers only:** no headlines, ids or per-impression rows. The
full-scale BehaviorGPT inference has **not** been run.

## 1. Phase 0 probes on `retail_catalog_500k` (shared catalog, no MIND data)

157 ranking calls in total, all sent without a user id.

| Probe | Result | Consequence |
|---|---|---|
| `id` filter with 100 ids | All 100 returned, finite scores (range 1.14–1.39) | Chunks of ≤ 100 ids |
| 120 ids, `limit=100` | 100 returned, no error | Never send > 100: the rest would be dropped silently |
| Same history, two overlapping chunks | Overlap scores differ by ≤ 0.00032 (score SD over items: 0.041) | Chunks of one history share one scale |
| Identical call twice | Max difference 0.00032, median 0.00007, Spearman 0.99994 | Small nondeterminism; the response cache makes runs reproducible |
| Unknown id in the filter | Dropped silently (9 of 10 returned) | The wrapper counts missing ids; the decoder ranks them last |
| Latency, 1 worker (12 calls) | 3.0 calls/s, median 0.22 s | — |
| 4 / 8 / 16 workers (24 / 24 / 32 calls) | 12.5 / 19.1 / 29.9 calls/s; median 0.29 / 0.37 / 0.41 s; **no 429 or error** | Default 8 workers, throttle 15 calls/s, retries on 429/5xx |
| History size accepted | 200, 800 and 2,000 events: all OK, ~0.4–0.5 s | Request size is not a limit |
| **Effective context** | A signal session at the start of the history moves candidate scores by 0.32 with 7 events, 0.07 with 265 events, and **0** with 277 or more (bit-identical scores) | BehaviorGPT reads about the last 256–270 events. **Cap *K* = 256 events**, current session included, keeping the most recent sessions whole |

### V3 go/no-go (`predict="purchase"` on the current session's own viewed items)

The rule was fixed before the run. GO only if all three hold:

- **(a)** the own items come back with finite scores;
- **(b)** the history moves the within-screen ranking the right way: for both category pairs,
  P(X above Y | history of X) − P(X above Y | history of Y) ≥ 0.2;
- **(c)** median Spearman ρ across shuffled view orders ≥ 0.5.

| Check | lego vs lipstick | running shoes vs coffee makers | Pass? |
|---|---|---|---|
| (a) own items returned, finite | yes | yes | yes |
| (b) history shift | **−0.12** | **−0.05** | **no** |
| (c) order stability (median ρ) | 0.89 | 0.36 | yes (overall median 0.68) |

**Decision: V3 is NO-GO and is dropped** (plan L6). Two side observations:

- Own items were not suppressed when scored through the `id` filter. Their mean rank among
  20 was 5.5–7.6, above the same-category controls.
- With an earlier session of category X, the purchase ranking of the screen did not favour X.

## 2. Phase 1: MIND-small facts (plan §3.5)

| Fact | Training file | Validation file |
|---|---|---|
| Impressions | 156,965 (2019-11-09 … 11-14) | 73,152 (2019-11-15) |
| Persons | 50,000 | 50,000 |
| Headlines shown per impression | mean 37.2, median 24, max 299 | mean 37.5, median 23, max 295 |
| Impressions with > 100 shown (chunked) | 12,669 | 5,967 |
| Clicks per impression | mean 1.51, median 1, max 35 | mean 1.52, median 1, max 24 |
| Impressions without a click / all clicked | 0 / 0 | 0 / 0 |
| History length | mean 32.5, median 19, max 558 | mean 32.3, median 19, max 444 |
| Impressions with an empty History | 3,238 | 2,214 |

- **News.** 65,238 articles (≤ 100,000: one catalog). No presented or History id is missing
  from the news files. No duplicate entries in any impression.
- **History.** The History field is constant per person, and identical across the two files.
- **Earlier impressions.** Only **12.3 % of validation impressions** belong to persons with any
  training-file impression. On the training file's last day (the selection pool), **83.6 %**
  of impressions have earlier impressions.

### Record sets (fixed seed)

| Set | Records | Persons | With earlier impressions | History clicks only | No previous decisions | *k* = 1 | > 100 shown |
|---|---|---|---|---|---|---|---|
| Evaluation (validation file) | 5,000 | 4,833 | 594 (11.9 %) | 4,273 (85.5 %) | 133 (2.7 %) | 4,752 (95.0 %) | 396 |
| Selection (training, last day) | 1,000 | 981 | 821 (82.1 %) | 176 (17.6 %) | 3 (0.3 %) | 638 (63.8 %) | 97 |
| Probe (training, last day) | 20 | 20 | 19 | 1 | 0 | 14 | 0 |

- **The *k* rule on the selection records.** Mean *k* is 2.06 against 1.53 true clicks. *k*
  equals the true count in 57 % of episodes; mean absolute error 1.06.
- **Re-presented observations** (seen in an earlier episode): 0.6 % of evaluation and 4.7 % of
  selection observations.
- **History after the cap.** V2 histories average 120 events on the selection records, and
  21.6 % of them are cut at *K*.

## 3. Phase 3: the MIND headline catalog

| Item | Result |
|---|---|
| Trial slice | 300 items, `ready` in 6 s, 0 rejected |
| Full catalog (same name, so the slice was refreshed and one slot is used) | 65,238 items, `ready` in 416 s, 0 rejected, text-only, `reference_time` 2019-11-15 |
| `similar_items` spot check (20 items × 10 neighbours) | Same MIND category 88.0 % (14.5 % by chance); same subcategory 60.5 % (3.9 % by chance) |
| Deletion dry run | Lists `.temp/data`, `.temp/work` and the owned catalog `mind_small_headlines`; nothing else |

## 4. Small API checks on the MIND catalog (20 probe records)

The probe records are disjoint from the selection and evaluation records.

| Run | Requests | Retries | Missing ids | Time (8 workers) |
|---|---|---|---|---|
| V1, V1o, V2 | 28 each (V2: 1 cache hit shared with V1o) | 0 | 0 | ~2.5 s each |
| V3 (for the record; no-go) | 20 | 0 | 0 | — |
| V1 with empty history | 28 | 0 | 0 | — |

- **What was checked.** The pipeline runs end to end, and every presented headline comes back
  scored.
- **Metric values are not reported.** On 20 records they carry no information: even the
  random baseline scored an AUC of 0.65 there, about 2 standard errors above 0.5.

## 5. Request budget for the full-scale run (exact)

Counted from *k* and the chunk count of every record by `scripts/04_predict.py estimate`.

| Run | Ranking requests |
|---|---|
| Selection: 1,000 records × V1, V1o, V2 (2,442 each) | 7,326 |
| Evaluation: 5,000 records, chosen variant | 6,511 |
| Evaluation with empty history (E3a) | 6,511 |
| **Total** | **20,348** |

- **Time.** At the default throttle of 15 calls/s this is about 23 minutes of calls, and about
  17 minutes at 8 workers without the throttle.
- **Before this run** (for comparison):
  - Phase 0: 157 ranking calls on `retail_catalog_500k`;
  - MIND probes: 132 ranking calls and 20 `similar_items` calls;
  - 2 catalog uploads.
