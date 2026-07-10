"""Command-line interface for the Step 8 Editorial MR.Q experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.editorial_mrq.oracle import (
    export_swapped_oracle_prompts,
    score_swapped_oracle_predictions,
)
from preference_futures.editorial_mrq.runtime import (
    prepare_editorial_mrq,
    run_editorial_embeddings,
    run_editorial_rankers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    swap = subparsers.add_parser(
        "oracle-swap",
        help="export candidate-swapped oracle prompts without opening the answer key",
    )
    swap.add_argument("--prompts", type=Path, required=True)
    swap.add_argument("--output-dir", type=Path, required=True)
    swap.add_argument("--force", action="store_true")

    score = subparsers.add_parser(
        "oracle-score-swap",
        help="score original and swapped oracle predictions",
    )
    score.add_argument("--original-predictions", type=Path, required=True)
    score.add_argument("--swapped-predictions", type=Path, required=True)
    score.add_argument("--answer-key", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser(
        "prepare",
        help="snapshot the frozen embedder and freeze the Step 8 contract",
    )
    prepare.add_argument("--training-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--model-id",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    prepare.add_argument("--model-revision", default="main")
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--max-length", type=int, default=384)
    prepare.add_argument("--embedding-batch-size", type=int, default=24)
    prepare.add_argument("--ranker-batch-size", type=int, default=256)
    prepare.add_argument("--maximum-epochs", type=int, default=100)
    prepare.add_argument("--learning-rate", type=float, default=1e-3)
    prepare.add_argument("--weight-decay", type=float, default=1e-4)
    prepare.add_argument("--patience", type=int, default=12)
    prepare.add_argument("--hidden-size", type=int, default=256)
    prepare.add_argument("--bottleneck-size", type=int, default=64)
    prepare.add_argument("--dropout", type=float, default=0.1)
    prepare.add_argument("--teacher-weight", type=float, default=0.25)
    prepare.add_argument("--force", action="store_true")

    embed = subparsers.add_parser(
        "embed",
        help="extract frozen context and candidate embeddings",
    )
    embed.add_argument("--editorial-dir", type=Path, required=True)
    embed.add_argument("--device", default="auto")
    embed.add_argument("--force", action="store_true")

    train = subparsers.add_parser(
        "train",
        help="train symmetric linear and tiny MR.Q preference rankers",
    )
    train.add_argument("--editorial-dir", type=Path, required=True)
    train.add_argument("--folds", default="all")
    train.add_argument("--rankers", default="all")
    train.add_argument("--teacher-predictions", type=Path)
    train.add_argument("--device", default="auto")
    train.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "oracle-swap":
        manifest = export_swapped_oracle_prompts(
            prompts_path=args.prompts,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 8.0 swapped oracle prompts exported.")
        print(f"  Items:   {manifest['items']}")
        print(f"  Prompts: {manifest['swapped_prompts_path']}")
        return 0

    if args.command == "oracle-score-swap":
        report = score_swapped_oracle_predictions(
            original_predictions_path=args.original_predictions,
            swapped_predictions_path=args.swapped_predictions,
            answer_key_path=args.answer_key,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 8.0 candidate-swap oracle score complete.")
        print(f"  Original accuracy: {report['original']['accuracy']:.6f}")
        print(f"  Swapped accuracy:  {report['swapped_translated']['accuracy']:.6f}")
        print(f"  Order consistency: {report['order_consistency']['rate']:.6f}")
        print(
            "  Consistent accuracy: "
            f"{report['consistent_subset']['accuracy']:.6f}"
        )
        return 0

    if args.command == "prepare":
        contract = prepare_editorial_mrq(
            training_directory=args.training_dir,
            output_directory=args.output_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            seed=args.seed,
            maximum_sequence_length=args.max_length,
            embedding_batch_size=args.embedding_batch_size,
            ranker_batch_size=args.ranker_batch_size,
            maximum_epochs=args.maximum_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            hidden_size=args.hidden_size,
            bottleneck_size=args.bottleneck_size,
            dropout=args.dropout,
            teacher_weight=args.teacher_weight,
            force=args.force,
        )
        print("Step 8 Editorial MR.Q contract prepared.")
        print(f"  Embedder: {contract['embedder']['model_id']}")
        print(f"  Revision: {contract['embedder']['resolved_revision']}")
        print(f"  Contract: {args.output_dir.expanduser().resolve() / 'contract.json'}")
        return 0

    if args.command == "embed":
        report = run_editorial_embeddings(
            args.editorial_dir,
            device=args.device,
            force=args.force,
        )
        print("Step 8 frozen embeddings complete.")
        print(f"  Rows:        {report['rows']}")
        print(f"  Hidden size: {report['hidden_size']}")
        print(f"  Device:      {report['device']}")
        return 0

    summary = run_editorial_rankers(
        args.editorial_dir,
        folds=args.folds,
        rankers=args.rankers,
        teacher_predictions_path=args.teacher_predictions,
        device=args.device,
        force=args.force,
    )
    print("Step 8 Editorial MR.Q training complete.")
    print(f"  Completed: {len(summary['completed'])}")
    print(f"  Skipped:   {len(summary['skipped'])}")
    for result in summary["completed"]:
        print(
            f"  fold {result['fold']:02d} / {result['ranker']}: "
            f"accuracy={result['test_accuracy']:.6f}, "
            f"gate={result['source_gate_passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
