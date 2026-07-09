"""CLI for independent verification of persisted Step 2 corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from preference_futures.corpora.verify import (
    render_corpus_verification_markdown,
    verify_compute_matched_corpora,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-verify-corpora")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    report = verify_compute_matched_corpora(output)
    (output / "corpus-verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "corpus-verification.md").write_text(
        render_corpus_verification_markdown(report), encoding="utf-8"
    )
    print("Compute-matched corpus verification complete.")
    print(f"  Status: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  Output: {output}")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    for error in report["errors"]:
        print(f"  ERROR: {error}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
