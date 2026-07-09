"""Command line entry point for Step 3 fixed-budget training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.training.runtime import prepare_training, run_training_jobs
from preference_futures.training.verify import verify_training_runs, write_training_verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="snapshot the base model and freeze Step 3")
    prepare.add_argument("--corpora-dir", type=Path, required=True)
    prepare.add_argument("--episodes", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model-id", default="distilbert/distilbert-base-uncased")
    prepare.add_argument("--model-revision", default="main")
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--max-length", type=int, default=256)
    prepare.add_argument("--batch-size", type=int, default=16)
    prepare.add_argument("--update-steps", type=int, default=600)
    prepare.add_argument("--learning-rate", type=float, default=2e-5)
    prepare.add_argument("--weight-decay", type=float, default=0.01)
    prepare.add_argument("--warmup-steps", type=int, default=60)
    prepare.add_argument("--gradient-clip-norm", type=float, default=1.0)
    prepare.add_argument("--log-every-steps", type=int, default=25)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="train selected frozen fold/regime jobs")
    run.add_argument("--training-dir", type=Path, required=True)
    run.add_argument("--folds", default="all")
    run.add_argument("--regimes", default="all")
    run.add_argument("--device", default="auto")
    run.add_argument("--smoke-steps", type=int)
    run.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify", help="verify persisted Step 3 runs")
    verify.add_argument("--training-dir", type=Path, required=True)
    verify.add_argument("--folds", default="all")
    verify.add_argument("--regimes", default="all")
    verify.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        contract = prepare_training(
            corpora_directory=args.corpora_dir,
            episodes_path=args.episodes,
            output_directory=args.output_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            seed=args.seed,
            maximum_sequence_length=args.max_length,
            batch_size=args.batch_size,
            update_steps=args.update_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_clip_norm=args.gradient_clip_norm,
            log_every_steps=args.log_every_steps,
            force=args.force,
        )
        print("Step 3 training contract prepared.")
        print(f"  Contract: {args.output_dir.expanduser().resolve() / 'contract.json'}")
        print(f"  Model:    {contract['model']['model_id']}")
        print(f"  Revision: {contract['model']['resolved_revision']}")
        print(f"  Jobs:     {contract['expected_training_jobs']}")
        return 0

    if args.command == "run":
        summary = run_training_jobs(
            args.training_dir,
            folds=args.folds,
            regimes=args.regimes,
            device=args.device,
            smoke_steps=args.smoke_steps,
            force=args.force,
        )
        print("Step 3 training command complete.")
        print(f"  Completed: {len(summary['completed_jobs'])}")
        print(f"  Skipped:   {len(summary['skipped_jobs'])}")
        print(f"  Device:    {summary['device']}")
        print(f"  Output:    {summary['run_root']}")
        return 0

    report = verify_training_runs(
        args.training_dir,
        folds=args.folds,
        regimes=args.regimes,
        smoke=args.smoke,
    )
    write_training_verification(args.training_dir, report)
    print(f"Step 3 training verification: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  Expected jobs: {report['observed']['expected_jobs']}")
    print(f"  Observed jobs: {report['observed']['observed_jobs']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
