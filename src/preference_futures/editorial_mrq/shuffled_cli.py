"""Command-line interface for the final shuffled-preference MR.Q control."""

from __future__ import annotations

import argparse
from pathlib import Path

from preference_futures.editorial_mrq.shuffled_aggregate import aggregate_shuffled_control
from preference_futures.editorial_mrq.shuffled_prepare import prepare_shuffled_control
from preference_futures.editorial_mrq.shuffled_runtime import (
    run_shuffled_future_probes,
    train_shuffled_source_models,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq-shuffled")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--editorial-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train-source")
    train.add_argument("--control-dir", type=Path, required=True)
    train.add_argument("--replicas", default="all")
    train.add_argument("--folds", default="all")
    train.add_argument("--device", default="auto")
    train.add_argument("--force", action="store_true")

    transfer = subparsers.add_parser("run-transfer")
    transfer.add_argument("--control-dir", type=Path, required=True)
    transfer.add_argument("--replicas", default="all")
    transfer.add_argument("--folds", default="all")
    transfer.add_argument("--arms", default="all")
    transfer.add_argument("--device", default="auto")
    transfer.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--control-dir", type=Path, required=True)
    aggregate.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_shuffled_control(
            args.editorial_dir,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 8.7 shuffled-control contract prepared.")
        print(f"  Replicas: {report['shuffle_replicates']}")
        print(f"  Contract: {Path(report['output_directory']) / 'contract.json'}")
        return 0
    if args.command == "train-source":
        report = train_shuffled_source_models(
            args.control_dir,
            replicas=args.replicas,
            folds=args.folds,
            device=args.device,
            force=args.force,
        )
        print("Step 8.7 shuffled source training complete.")
        print(f"  Completed: {len(report['completed'])}")
        print(f"  Skipped:   {len(report['skipped'])}")
        return 0
    if args.command == "run-transfer":
        report = run_shuffled_future_probes(
            args.control_dir,
            replicas=args.replicas,
            folds=args.folds,
            arms=args.arms,
            device=args.device,
            force=args.force,
        )
        print("Step 8.7 shuffled future probes complete.")
        print(f"  Completed: {len(report['completed'])}")
        print(f"  Skipped:   {len(report['skipped'])}")
        return 0

    report = aggregate_shuffled_control(args.control_dir, force=args.force)
    specificity = report["authentic_preference_specificity"]
    print("Step 8.7 shuffled-control aggregation complete.")
    for arm in ("mrq_choice_aware", "mrq_blind"):
        comparison = report["comparisons"]["mean"][arm]
        interval = comparison["confidence_interval_95"]
        print(
            f"  {arm}: difference={comparison['mean_log_loss_difference']:.6f} "
            f"[{interval[0]:.6f}, {interval[1]:.6f}], "
            f"negative_replicates={report['comparisons']['negative_point_estimate_counts'][arm]}/"
            f"{report['shuffle_replicates']}"
        )
    print(f"  Authentic preference specificity: {specificity['supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
