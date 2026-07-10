"""Command-line interface for Step 8.4 future-transfer experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.editorial_mrq.transfer import (
    aggregate_future_transfer,
    prepare_future_transfer,
    run_future_transfer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq-transfer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze the Step 8.4 transfer contract")
    prepare.add_argument("--editorial-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="run identical future probes for transfer arms")
    run.add_argument("--transfer-dir", type=Path, required=True)
    run.add_argument("--folds", default="all")
    run.add_argument("--arms", default="all")
    run.add_argument("--device", default="auto")
    run.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="pool held-out transfer predictions")
    aggregate.add_argument("--transfer-dir", type=Path, required=True)
    aggregate.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        contract = prepare_future_transfer(
            args.editorial_dir,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 8.4 future-transfer contract prepared.")
        print(f"  Arms:     {len(contract['arms'])}")
        print(f"  Contract: {Path(contract['output_directory']) / 'contract.json'}")
        return 0

    if args.command == "run":
        summary = run_future_transfer(
            args.transfer_dir,
            folds=args.folds,
            arms=args.arms,
            device=args.device,
            force=args.force,
        )
        print("Step 8.4 future probes complete.")
        print(f"  Completed: {len(summary['completed'])}")
        print(f"  Skipped:   {len(summary['skipped'])}")
        for result in summary["completed"]:
            print(
                f"  fold {result['fold']:02d} / {result['arm']}: "
                f"log_loss={result['test_log_loss']:.6f}, "
                f"accuracy={result['test_accuracy']:.6f}"
            )
        return 0

    report = aggregate_future_transfer(args.transfer_dir, force=args.force)
    print("Step 8.4 future-transfer aggregation complete.")
    for arm, values in report["arms"].items():
        metrics = values["pooled_test"]
        print(
            f"  {arm}: log_loss={metrics['log_loss']:.6f}, "
            f"accuracy={metrics['accuracy']:.6f}"
        )
    primary = report["comparisons"]["primary"]
    interval = primary["confidence_interval_95"]
    print(
        "  Primary log-loss difference: "
        f"{primary['mean_log_loss_difference']:.6f} "
        f"[{interval[0]:.6f}, {interval[1]:.6f}]"
    )
    print(f"  Future transfer supported: {report['future_transfer']['supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
