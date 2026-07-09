"""Command-line entry point for Step 5 frozen representations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.representations.runtime import (
    prepare_representations,
    run_representation_jobs,
)
from preference_futures.representations.verify import (
    verify_representation_runs,
    write_representation_verification,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-representations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze the Step 5 extraction contract")
    prepare.add_argument("--selection-manifest", type=Path, required=True)
    prepare.add_argument("--training-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="extract selected frozen representation jobs")
    run.add_argument("--representation-dir", type=Path, required=True)
    run.add_argument("--folds", default="all")
    run.add_argument("--arms", default="all")
    run.add_argument("--device", default="auto")
    run.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify", help="verify persisted Step 5 matrices")
    verify.add_argument("--representation-dir", type=Path, required=True)
    verify.add_argument("--folds", default="all")
    verify.add_argument("--arms", default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        contract = prepare_representations(
            selection_manifest_path=args.selection_manifest,
            training_directory=args.training_dir,
            output_directory=args.output_dir,
            batch_size=args.batch_size,
            force=args.force,
        )
        print("Step 5 representation contract prepared.")
        print(f"  Jobs:       {contract['expected_extraction_jobs']}")
        print(f"  Matrices:   {contract['expected_partition_artifacts']}")
        print(f"  Pooling:    {contract['representation']['pooling']}")
        print(f"  Contract:   {args.output_dir.expanduser().resolve() / 'contract.json'}")
        return 0

    if args.command == "run":
        summary = run_representation_jobs(
            args.representation_dir,
            folds=args.folds,
            arms=args.arms,
            device=args.device,
            force=args.force,
        )
        print("Step 5 representation extraction complete.")
        print(f"  Completed: {len(summary['completed_jobs'])}")
        print(f"  Skipped:   {len(summary['skipped_jobs'])}")
        print(f"  Device:    {summary['device']}")
        print(f"  Output:    {summary['run_root']}")
        return 0

    report = verify_representation_runs(
        args.representation_dir,
        folds=args.folds,
        arms=args.arms,
    )
    write_representation_verification(args.representation_dir, report)
    print(f"Step 5 representation verification: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  Expected jobs: {report['observed']['expected_jobs']}")
    print(f"  Observed jobs: {report['observed']['observed_jobs']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
