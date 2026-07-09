"""CLI for Step 2 compute-matched corpus construction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.corpora.build import (
    build_compute_matched_corpora,
    load_json,
    load_jsonl,
    write_compute_matched_corpora,
)
from preference_futures.corpora.temporal import (
    extract_independent_temporal_pairs,
    write_temporal_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-corpora")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", default="nyt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temporal-max-articles", type=int, default=20000)
    parser.add_argument("--temporal-pool-multiplier", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes_path = args.episodes.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    split_manifest_path = splits_dir / "manifest.json"

    episodes = load_jsonl(episodes_path)
    split_manifest = load_json(split_manifest_path)
    outer_folds = int(split_manifest["outer_folds"])
    fold_documents = {
        fold: load_json(splits_dir / f"fold-{fold:02d}.json")
        for fold in range(outer_folds)
    }
    evaluation_lineages = {str(record["lineage_id"]) for record in episodes}
    target_pairs = max(
        len(episodes),
        int(round(len(episodes) * args.temporal_pool_multiplier)),
    )
    temporal_path = output_dir / "temporal-pairs.jsonl"
    temporal_audit_path = output_dir / "temporal-pairs-audit.json"
    temporal_pairs, temporal_audit = extract_independent_temporal_pairs(
        args.database,
        excluded_lineages=evaluation_lineages,
        source_name=args.source_name,
        target_pairs=target_pairs,
        seed=args.seed,
        max_articles=args.temporal_max_articles,
    )
    write_temporal_pairs(temporal_path, temporal_audit_path, temporal_pairs, temporal_audit)

    manifest, outputs = build_compute_matched_corpora(
        episodes,
        split_manifest,
        fold_documents,
        temporal_pairs,
        seed=args.seed,
        episodes_path=episodes_path,
        split_manifest_path=split_manifest_path,
        temporal_pairs_path=temporal_path,
    )
    manifest["temporal_pair_audit"] = temporal_audit
    write_compute_matched_corpora(output_dir, manifest, outputs)

    print("Compute-matched source corpora built.")
    print(f"  Episodes:       {len(episodes):,}")
    print(f"  Temporal pairs: {len(temporal_pairs):,}")
    print(f"  Outer folds:    {outer_folds}")
    print(f"  Output:         {output_dir}")
    print("  Gates:")
    for name, passed in manifest["gates"].items():
        print(f"    {'PASS' if passed else 'FAIL'}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
