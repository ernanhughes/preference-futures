"""Command-line interface for the Step 7 diagnostic gates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.diagnostics.labels import run_future_label_integrity_audit
from preference_futures.diagnostics.preference import (
    export_preference_oracle_sample,
    run_preference_learnability_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    learnability = subparsers.add_parser(
        "preference-learnability",
        help="run tiny-set memorization and extended-budget preference learning curves",
    )
    learnability.add_argument("--training-dir", type=Path, required=True)
    learnability.add_argument("--output-dir", type=Path, required=True)
    learnability.add_argument("--fold", type=int, default=0)
    learnability.add_argument("--budgets", default="600,1200,2400,5000,10000")
    learnability.add_argument("--memorization-sizes", default="256,512")
    learnability.add_argument("--memorization-steps", type=int, default=5000)
    learnability.add_argument("--train-evaluation-size", type=int, default=2048)
    learnability.add_argument("--device", default="auto")
    learnability.add_argument("--force", action="store_true")

    oracle = subparsers.add_parser(
        "oracle-export",
        help="export blinded preference prompts and a separate answer key",
    )
    oracle.add_argument("--training-dir", type=Path, required=True)
    oracle.add_argument("--output-dir", type=Path, required=True)
    oracle.add_argument("--fold", type=int, default=0)
    oracle.add_argument("--sample-size", type=int, default=300)
    oracle.add_argument("--seed", type=int, default=17)
    oracle.add_argument("--force", action="store_true")

    labels = subparsers.add_parser(
        "future-label-audit",
        help="quantify disagreement between alignment normalization and future labels",
    )
    labels.add_argument("--db", type=Path, required=True)
    labels.add_argument("--output-dir", type=Path, required=True)
    labels.add_argument("--table")
    labels.add_argument("--split-table")
    labels.add_argument("--source-name")
    labels.add_argument("--seed", type=int, default=0)
    labels.add_argument("--max-articles", type=int, default=0)
    labels.add_argument("--max-examples", type=int, default=0)
    labels.add_argument("--sources", default="")
    labels.add_argument("--context-before", type=int, default=1)
    labels.add_argument("--context-after", type=int, default=1)
    labels.add_argument("--min-sentence-chars", type=int, default=20)
    labels.add_argument("--max-sentence-chars", type=int, default=500)
    labels.add_argument("--min-edit-similarity", type=float, default=0.15)
    labels.add_argument("--max-edit-similarity", type=float, default=0.98)
    labels.add_argument("--sample-limit", type=int, default=50)
    labels.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preference-learnability":
        summary = run_preference_learnability_audit(
            training_directory=args.training_dir,
            output_directory=args.output_dir,
            fold=args.fold,
            budgets=_parse_ints(args.budgets),
            memorization_sizes=_parse_ints(args.memorization_sizes),
            memorization_steps=args.memorization_steps,
            train_evaluation_size=args.train_evaluation_size,
            device=args.device,
            force=args.force,
        )
        best = summary["best_validation_run"]
        print("Step 7A preference learnability audit complete.")
        print(f"  Outcome: {summary['outcome']}")
        print(
            "  Tiny-set memorization: "
            f"{summary['gates']['tiny_set_memorization_passed']}"
        )
        print(
            "  Learned above prior:   "
            f"{summary['gates']['validation_learned_above_prior']}"
        )
        print(
            "  Best validation:       "
            f"{best['accuracy']:.6f} at {best['update_steps']} updates"
        )
        print(
            "  Report:                "
            f"{args.output_dir.expanduser().resolve() / 'preference-learnability-summary.md'}"
        )
        return 0

    if args.command == "oracle-export":
        manifest = export_preference_oracle_sample(
            training_directory=args.training_dir,
            output_directory=args.output_dir,
            fold=args.fold,
            sample_size=args.sample_size,
            seed=args.seed,
            force=args.force,
        )
        print("Preference oracle sample exported.")
        print(f"  Items:      {manifest['sample_size']}")
        print(f"  Prompts:    {manifest['prompts_path']}")
        print(f"  Answer key: {manifest['answer_key_path']}")
        return 0

    report = run_future_label_integrity_audit(
        database_path=args.db,
        output_directory=args.output_dir,
        table=args.table,
        split_table=args.split_table,
        source_name=args.source_name,
        seed=args.seed,
        max_articles=args.max_articles,
        max_examples=args.max_examples,
        sources=tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        context_before=args.context_before,
        context_after=args.context_after,
        min_sentence_chars=args.min_sentence_chars,
        max_sentence_chars=args.max_sentence_chars,
        min_edit_similarity=args.min_edit_similarity,
        max_edit_similarity=args.max_edit_similarity,
        sample_limit=args.sample_limit,
        force=args.force,
    )
    counts = report["counts"]
    print("Step 7B future-label integrity audit complete.")
    print(f"  Episodes:        {counts['episodes']}")
    print(f"  Label mismatches:{counts['normalization_mismatch_labels']}")
    print(
        "  Report:          "
        f"{args.output_dir.expanduser().resolve() / 'future-label-integrity.md'}"
    )
    return 0


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
