"""Command-line interface for the confirmatory milestone snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.milestone.freeze import (
    DEFAULT_MILESTONE_NAME,
    freeze_confirmatory_milestone,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-milestone")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--encoder-selection-dir", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--repository-result-dir", type=Path, required=True)
    parser.add_argument("--milestone-name", default=DEFAULT_MILESTONE_NAME)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = freeze_confirmatory_milestone(
        repository_root=args.repository_root,
        splits_directory=args.splits_dir,
        corpora_directory=args.corpora_dir,
        training_directory=args.training_dir,
        encoder_selection_directory=args.encoder_selection_dir,
        representation_directory=args.representation_dir,
        probe_directory=args.probe_dir,
        release_root=args.release_root,
        repository_result_directory=args.repository_result_dir,
        milestone_name=args.milestone_name,
        allow_dirty=args.allow_dirty,
        force=args.force,
    )
    print("Confirmatory milestone frozen.")
    print(f"  Classification: {summary['classification']}")
    print(f"  Files:          {summary['file_count']}")
    print(f"  Archive:        {summary['archive_path']}")
    print(f"  Archive SHA256: {summary['archive_sha256']}")
    print(f"  Result record:  {summary['repository_result_markdown']}")
    print(f"  Suggested tag:  {summary['suggested_git_tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
