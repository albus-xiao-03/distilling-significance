# distilling-significance

A hackathon prototype. An unchanged behaviour model, Unbox AI's **BehaviorGPT**, predicts the
**attention decisions** people made on news headlines: which of the headlines shown to them they
chose to open. Those predictions become **pseudo-labels** for supervised learning, and they are
measured against the real decisions. The plan is [`plans/R1.md`](plans/R1.md).

> **Status:** the R1 pipeline is implemented and tested, the MIND headline catalog is
> uploaded, and small API checks pass ([`reports/feasibility.md`](reports/feasibility.md)).
> The full-scale BehaviorGPT run (20,348 ranking requests) waits for the owner's approval. No
> labels exist yet.

## How it works

1. **Data.** MIND-small impressions become decision episodes. Each holds the headlines shown
   to a person, the person's previous decisions (History clicks, and earlier impressions on
   earlier days), and their real clicks, kept apart for evaluation only.
2. **Catalog.** Every headline is a text-only item of a private Unbox AI catalog.
3. **Representation.** Previous decisions are written as BehaviorGPT events. There are three
   variants:
   - V1 views the attended headlines;
   - V1o adds an order of them;
   - V2 views everything shown and orders the attended headlines.

   The variant is chosen on 1,000 selection records.
4. **Prediction.** BehaviorGPT ranks the headlines shown as the person's next action, *k*
   times, re-ranking after each choice. *k* comes from the person's own past attention rate.
   The top *k* are the predicted attention decisions: the pseudo-labels.
5. **Evaluation.**
   - Against the real decisions: AUC, MRR, nDCG, F1 and κ, with bootstrap CIs over persons.
   - Against four baselines.
   - With and without history.
   - Distillation into a student classifier.

BehaviorGPT itself is never changed or trained.

## Running it

```sh
source .temp/env_setup/devenv.sh                 # project venv; every cache stays in .temp/
python -m pip install --no-deps -e .             # once: scripts run isolated (python -I)
python -I scripts/01_prepare.py                  # download + check MIND-small, stats, records
python -I scripts/02_build.py                    # catalog file, decision episodes, profile
python -I scripts/03_embed.py slice              # trial upload (300 headlines)
python -I scripts/03_embed.py full               # all headlines, same catalog
python -I scripts/03_embed.py check              # status and similar_items spot check
python -I scripts/04_predict.py estimate         # exact number of ranking requests
python -I scripts/04_predict.py run --set probe  # small check on 20 probe records
# full-scale inference, only with the owner's approval:
python -I scripts/04_predict.py run --set selection --approved
python -I scripts/04_predict.py select           # best mean AUC on the selection records
python -I scripts/04_predict.py run --set evaluation --variants chosen --approved
python -I scripts/04_predict.py run --set evaluation --variants chosen --empty-history --approved
python -I scripts/05_evaluate.py --set evaluation   # E1-E3 and baselines
python -I scripts/06_dataset.py                  # outputs/mind_attention_pseudolabels.parquet
python -I scripts/07_distil.py                   # E4
```

Notes:

- **Secrets.** `UNBOXAI_API_KEY` and `HF_TOKEN` are read from the git-ignored `.env` and never
  printed.
- **Cache.** Every API response is cached in `.temp/cache/`, so a stopped run resumes.
- **Guards.** Runs on the selection and evaluation records refuse to start without
  `--approved`. The evaluation and distillation scripts refuse the evaluation records until
  their pseudo-labels exist.
- **Plan and data card.** [`plans/R1.md`](plans/R1.md) is the plan. The data card is
  [`DATACARD.md`](DATACARD.md).

## Layout

```
src/significance/
  config.py            paths, fixed settings, .env loader
  data/mind.py         MIND-small download, checks, safe extraction, readers, statistics
  data/records.py      evaluation / selection / probe records
  catalog.py           headlines -> catalog parquet
  episodes.py          decision episodes, variants V1/V1o/V2/V3, history cap, k rule
  teacher/api.py       BehaviorGPT calls: id filter, chunks, cache, retries, throttle
  teacher/decode.py    next-action decoding -> predicted decisions
  runner.py            estimate, run, choose the variant
  baselines.py         random, popularity, recency, content similarity
  evaluate.py          metrics and person-bootstrap intervals
  experiment.py        assessing predictors, comparisons, strata
  dataset.py           the six-column output and its checks
  student.py           E4 distillation student
  deletion.py          the deletion term
scripts/               01_prepare ... 07_distil, delete_data.py
tests/                 unit tests: synthetic data and a fake BehaviorGPT client only
reports/               aggregate results only
```

Tests and lint:

```sh
source .temp/env_setup/devenv.sh
python -m pytest
ruff check . && ruff format --check .
```

## Rights notice

- **No data in this repository.** The repository contains no MIND data and no label files.
  Raw data, catalogs, decision episodes, caches, real decisions and pseudo-labels stay in
  git-ignored folders (`.temp/`, `outputs/`) and are never committed or pushed.
- **MIND belongs to Microsoft.** The Microsoft News Dataset (MIND) is licensed under the
  [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf):
  - use for non-commercial, non-revenue-generating research only;
  - **no distribution** of the dataset;
  - results may be published only if no material portion of the dataset is included.

  Anyone who runs this code must get MIND from the official source and accept its licence
  first. This project reads MIND-small from the gated Hugging Face copy `yjw1029/MIND`.
- **The code licence covers code only.** The MIT licence in [`LICENSE`](LICENSE) applies to
  the code in this repository, not to MIND or to any label derived from it.
- **The labels come from BehaviorGPT.** The pseudo-labels are BehaviorGPT's predictions, made
  through the Unbox AI API and subject to Unbox AI's terms. To make them, MIND headlines
  (title, category, subcategory, entity names) are uploaded to a **private** Unbox AI catalog
  under the project owner's API key. Calls are made without a user id, so nothing is recorded
  for any person on the account.
- **Only aggregates are published.** Committed documents, such as reports and the data card,
  hold the method and aggregate numbers only: no headlines, no user ids, no per-impression
  rows.

## Deletion term

**What is deleted**

| Item | Where |
|---|---|
| MIND downloads, extracted files and the Hugging Face cache | `.temp/data/` |
| Catalog files, decision episodes, record tables, real decisions (evaluation tables) and predictions | `.temp/data/`, `.temp/work/` |
| The BehaviorGPT response cache | `.temp/cache/` |
| Generated label datasets | `outputs/` |
| The project's own Unbox AI catalog (`mind_small_headlines`) | Unbox AI, under the owner's key |

Shared catalogs and catalogs this project did not create are never touched.

**When**

- on request of a rights holder (Microsoft for MIND, Unbox AI for its model outputs);
- if the licence or terms turn out not to allow this use;
- at the end of the project.

**How**

```sh
source .temp/env_setup/devenv.sh
python -I scripts/delete_data.py            # dry run: lists what would be deleted
python -I scripts/delete_data.py --confirm  # deletes it
```

Manual equivalent, if the script cannot run:

1. Delete the folders `.temp/data/`, `.temp/work/`, `.temp/cache/` and `outputs/`.
2. Delete the project's catalog with the Unbox AI SDK. List the key's catalogs, then delete the
   one that is `owned` and named `mind_small_headlines`:
   ```python
   from behaviorgpt import UnboxAIClient
   client = UnboxAIClient(market="us")
   [c for c in client.list_catalogs() if c.owned]   # find its catalog_id
   client.delete_catalog("cat_...")                  # cannot be undone
   ```

**Commitment:** label files are never redistributed.
