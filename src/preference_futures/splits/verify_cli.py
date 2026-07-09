"""Command-line verification for persisted grouped split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from preference_futures.splits.verify import (
    render_grouped_split_verification_markdown,
    verify_grouped_split_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-verify-splits")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = verify_grouped_split_manifest(manifest)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(
        render_grouped_split_verification_markdown(report),
        encoding="utf-8",
    )

    print("Grouped split verification complete.")
    print(f"  Status:   {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  Manifest: {manifest_path}")
    print(f"  JSON:     {args.json_out.expanduser().resolve()}")
    print(f"  Markdown: {args.markdown_out.expanduser().resolve()}")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
