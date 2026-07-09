"""Command-line entry point for Step 4 encoder selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from preference_futures.selection.diagnostics import freeze_encoder_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-select-encoders")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = freeze_encoder_selection(args.training_dir, args.output_dir)
    print("Step 4 encoder selection complete.")
    print(f"  Entries:  {manifest['counts']['entries']}")
    print(f"  Eligible: {manifest['counts']['eligible_entries']}")
    print(f"  Output:   {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
