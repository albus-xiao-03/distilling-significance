"""Delete the project's MIND-derived data and its Unbox AI catalog (README "Deletion term").

    source .temp/env_setup/devenv.sh
    python -I scripts/delete_data.py            # dry run: lists what would be deleted
    python -I scripts/delete_data.py --confirm  # deletes it; cannot be undone

Deletes .temp/data/, .temp/work/, .temp/cache/, outputs/ and the catalog this API key owns
that is named mind_small_headlines. Shared catalogs and catalogs with other names are never
touched. Without UNBOXAI_API_KEY the local folders are still handled and the catalog step
is reported as skipped.
"""

import argparse

from significance import config, deletion


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete the project's data (dry run by default).")
    ap.add_argument("--confirm", action="store_true", help="really delete")
    args = ap.parse_args()
    p = config.paths()
    config.load_env()
    client = None
    try:
        config.require_secret("UNBOXAI_API_KEY")
    except RuntimeError:
        pass
    else:
        from behaviorgpt import UnboxAIClient

        client = UnboxAIClient(market=config.MARKET)
    report = deletion.run(p.root, client, confirm=args.confirm)
    for line in report.lines():
        print(line)
    if not args.confirm:
        print("dry run: nothing was deleted; pass --confirm to delete")


if __name__ == "__main__":
    main()
