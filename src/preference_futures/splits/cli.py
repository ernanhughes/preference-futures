"""Command-line interface for deterministic grouped split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.audit.report import load_episode_records
from preference_futures.splits.build import (
    build_grouped_split_manifest,
    load_numeric_flags,
    write_grouped_split_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-splits")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--numeric-flags", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes_path = args.episodes.expanduser().resolve()
    numeric_flags_path = (
        args.numeric_flags.expanduser().resolve() if args.numeric_flags is not None else None
    )

    records = load_episode_records(episodes_path)
    numeric_flags = load_numeric_flags(numeric_flags_path) if numeric_flags_path else {}
    manifest, folds = build_grouped_split_manifest(
        records,
        numeric_flags=numeric_flags,
        folds=args.folds,
        seed=args.seed,
        episodes_path=episodes_path,
        numeric_flags_path=numeric_flags_path,
    )
    write_grouped_split_artifacts(args.output_dir, manifest, folds)

    totals = manifest["totals"]
    print("Grouped split manifests complete.")
    print(f"  Episodes:       {totals['episodes']}")
    print(f"  Lineages:       {totals['lineages']}")
    print(f"  Outer folds:    {manifest['outer_folds']}")
    print(f"  Seed:           {manifest['seed']}")
    print(f"  Output:         {args.output_dir.expanduser().resolve()}")
    for name, passed in manifest["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    return 0 if all(manifest["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
