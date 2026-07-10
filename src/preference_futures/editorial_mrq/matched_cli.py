"""CLI for Step 8.6 matched generic controls."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from preference_futures.editorial_mrq.matched_aggregate import aggregate
from preference_futures.editorial_mrq.matched_runtime import prepare, run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq-matched-controls")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--transfer-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path)
    prepare_parser.add_argument("--force", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--matched-dir", type=Path, required=True)
    run_parser.add_argument("--folds", default="all")
    run_parser.add_argument("--arms", default="all")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--force", action="store_true")
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--matched-dir", type=Path, required=True)
    aggregate_parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        contract = prepare(
            args.transfer_dir,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 8.6 matched-control contract prepared.")
        print(f"  Controls: {len(contract['arms'])}")
        print(f"  Contract: {Path(contract['output_directory']) / 'contract.json'}")
        return 0
    if args.command == "run":
        summary = run(
            args.matched_dir,
            folds=args.folds,
            arms=args.arms,
            device=args.device,
            force=args.force,
        )
        print("Step 8.6 matched controls complete.")
        print(f"  Completed: {len(summary['completed'])}")
        print(f"  Skipped:   {len(summary['skipped'])}")
        for result in summary["completed"]:
            print(
                f"  fold {result['fold']:02d} / {result['arm']}: "
                f"log_loss={result['test_log_loss']:.6f}, "
                f"L2={result['selected_l2_lambda']:g}"
            )
        return 0
    report = aggregate(args.matched_dir, force=args.force)
    print("Step 8.6 matched-control aggregation complete.")
    for name in ("primary_dimension", "primary_regularisation"):
        comparison = report["comparisons"][name]
        interval = comparison["confidence_interval_95"]
        print(
            f"  {name}: {comparison['mean_log_loss_difference']:.6f} "
            f"[{interval[0]:.6f}, {interval[1]:.6f}]"
        )
    print(
        "  Specificity supported: "
        f"{report['compression_and_regularisation_specificity']['supported']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
