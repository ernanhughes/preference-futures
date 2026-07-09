"""Command-line interface for Step 2 compute-matched corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from preference_futures.audit.report import load_episode_records
from preference_futures.corpora.build import build_training_corpora, write_training_corpora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-corpora")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes_path = args.episodes.expanduser().resolve()
    split_manifest_path = args.split_manifest.expanduser().resolve()

    records = load_episode_records(episodes_path)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    manifest, corpora = build_training_corpora(
        records,
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=split_manifest_path,
        seed=args.seed,
    )
    write_training_corpora(args.output_dir, manifest, corpora)

    totals = manifest["totals"]
    print("Compute-matched training corpora complete.")
    print(f"  Episodes:    {totals['episodes']}")
    print(f"  Lineages:    {totals['lineages']}")
    print(f"  Corpora:     {len(manifest['corpus_names'])}")
    print(f"  Outer folds: {manifest['outer_folds']}")
    print(f"  Output:      {args.output_dir.expanduser().resolve()}")
    for name, passed in manifest["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    return 0 if all(manifest["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
