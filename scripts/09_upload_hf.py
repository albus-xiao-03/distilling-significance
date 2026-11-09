"""Upload the dataset to the owner's PRIVATE Hugging Face dataset repository.

    source .temp/env_setup/devenv.sh
    python -I scripts/09_upload_hf.py             # dry run: checks, lists what would go up
    python -I scripts/09_upload_hf.py --confirm   # uploads, then verifies every file

Before anything goes up it checks that:

- the dataset files pass their schema and label checks;
- the parquet files match the checksums in metadata.json;
- the token in .env (HF_TOKEN, never printed) can write;
- the target repository exists and is PRIVATE. A public repository is refused, because the
  MIND licence forbids distribution.

Only README.md (the dataset card, rendered from metadata.json), train.parquet,
validation.parquet and metadata.json are uploaded, never any other file of the project. The
upload goes over plain HTTPS, not the Xet protocol, whose host the sandbox does not allow.
"""

import argparse
import json
import sys

import pyarrow.parquet as pq

from significance import config, dataset, publish


def main() -> None:
    ap = argparse.ArgumentParser(description="Private Hugging Face copy of the dataset")
    ap.add_argument("--repo-id", default=config.HF_DATASET_REPO)
    ap.add_argument("--confirm", action="store_true", help="really upload")
    args = ap.parse_args()
    p = config.paths()
    folder = p.dataset

    meta = json.loads((folder / dataset.METADATA_NAME).read_text())
    for facts in meta["splits"].values():
        path = folder / facts["file"]
        dataset.validate_output(pq.read_table(path))
        if dataset.sha256_of(path) != facts["sha256"]:
            sys.exit(f"{path.name}: checksum differs from metadata.json")
        print(f"checked {path.name}: {facts['records']:,} records, {facts['rows']:,} rows")
    (folder / "README.md").write_text(publish.render_card(meta))
    print("dataset card written: dataset/README.md")

    import huggingface_hub.constants
    from huggingface_hub import HfApi, whoami

    huggingface_hub.constants.HF_HUB_DISABLE_XET = True
    config.load_env()
    token = config.require_secret("HF_TOKEN")
    who = whoami(token=token)
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    print(f"Hugging Face account: {who.get('name')} (token role: {role}; value not shown)")
    api = HfApi(token=token)
    try:
        info = publish.check_private(api, args.repo_id)
    except publish.PublishError as exc:
        sys.exit(f"refused: {exc}")
    print(f"target: {args.repo_id} (private: {info.private})")
    for name in publish.UPLOAD_FILES:
        print(f"  would upload: {name} ({(folder / name).stat().st_size:,} bytes)")
    if not args.confirm:
        print("dry run: nothing uploaded; pass --confirm to upload")
        return

    publish.upload(
        api,
        folder,
        args.repo_id,
        f"Upload the MIND attention pseudo-labels ({meta['records']:,} impressions)",
    )
    checks = publish.verify(api, folder, args.repo_id)
    for name, ok in checks.items():
        print(f"  verified {name}: {'identical' if ok else 'MISMATCH'}")
    if not all(checks.values()):
        sys.exit("verification failed")
    print(f"uploaded and verified: https://huggingface.co/datasets/{args.repo_id} (private)")


if __name__ == "__main__":
    main()
