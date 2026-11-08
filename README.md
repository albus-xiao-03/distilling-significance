# distilling-significance

A hackathon prototype. An unchanged behaviour model, Unbox AI's **BehaviorGPT**, predicts the
**attention decisions** people made on news headlines: which of the headlines shown to them they
chose to open. Those predictions become **pseudo-labels** for supervised learning, and they are
measured against the real decisions. The plan is [`plans/R1.md`](plans/R1.md).

> **Status:** R1 is being implemented. This README holds the mandatory **rights notice** and
> **deletion term** (plan §7.4). Usage instructions are added when the pipeline is complete.

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
