"""Command-line entry point for Step 6 identical future probes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.probes.runtime import prepare_probes, run_probe_jobs
from preference_futures.probes.verify import verify_probe_runs, write_probe_verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-probes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze the Step 6 probe contract")
    prepare.add_argument("--representation-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="train selected identical future probes")
    run.add_argument("--probe-dir", type=Path, required=True)
    run.add_argument("--folds", default="all")
    run.add_argument("--arms", default="all")
    run.add_argument("--device", default="auto")
    run.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify", help="verify and aggregate persisted probes")
    verify.add_argument("--probe-dir", type=Path, required=True)
    verify.add_argument("--folds", default="all")
    verify.add_argument("--arms", default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        contract = prepare_probes(
            representation_directory=args.representation_dir,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 6 future-probe contract prepared.")
        print(f"  Jobs:       {contract['expected_probe_jobs']}")
        print(f"  L2 grid:    {contract['probe']['l2_grid']}")
        print(f"  Primary:    {contract['metrics']['primary']}")
        print(f"  Contract:   {args.output_dir.expanduser().resolve() / 'contract.json'}")
        return 0

    if args.command == "run":
        summary = run_probe_jobs(
            args.probe_dir,
            folds=args.folds,
            arms=args.arms,
            device=args.device,
            force=args.force,
        )
        print("Step 6 future-probe training complete.")
        print(f"  Completed: {len(summary['completed_jobs'])}")
        print(f"  Skipped:   {len(summary['skipped_jobs'])}")
        print(f"  Device:    {summary['device']}")
        print(f"  Output:    {summary['run_root']}")
        return 0

    report, summary = verify_probe_runs(
        args.probe_dir,
        folds=args.folds,
        arms=args.arms,
    )
    write_probe_verification(args.probe_dir, report, summary)
    print(f"Step 6 future-probe verification: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  Expected jobs: {report['observed']['expected_jobs']}")
    print(f"  Observed jobs: {report['observed']['observed_jobs']}")
    if summary is not None and summary.get("status") == "complete":
        comparison = summary["paired_lineage_bootstrap"][
            "authentic_preference_vs_generic"
        ]
        lower, upper = comparison["confidence_interval_95"]
        print(
            "  Authentic vs generic log-loss improvement: "
            f"{comparison['log_loss_improvement']:+.6f} "
            f"(95% CI {lower:+.6f}, {upper:+.6f})"
        )
    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
