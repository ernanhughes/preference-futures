"""Command-line interface for the numeric shortcut audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from preference_futures.audit.numeric import (
    build_numeric_shortcut_report,
    render_numeric_shortcut_markdown,
)
from preference_futures.audit.report import load_episode_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-numeric-audit")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--flags-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes_path = args.episodes.expanduser().resolve()
    records = load_episode_records(episodes_path)
    report, flags = build_numeric_shortcut_report(records, source_path=episodes_path)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(
        render_numeric_shortcut_markdown(report),
        encoding="utf-8",
    )

    args.flags_out.parent.mkdir(parents=True, exist_ok=True)
    with args.flags_out.open("w", encoding="utf-8", newline="\n") as stream:
        for record in flags:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    categories = report["categories"]
    print("Numeric shortcut audit complete.")
    print(f"  Episodes:              {report['dataset']['episodes']}")
    print(f"  Number changed:        {categories['number_changed']['episodes']}")
    print(f"  Number dominant:       {categories['number_dominant_edit']['episodes']}")
    print(f"  Casualty-count update: {categories['casualty_count_update']['episodes']}")
    print(f"  JSON:                  {args.json_out.expanduser().resolve()}")
    print(f"  Markdown:              {args.markdown_out.expanduser().resolve()}")
    print(f"  Flags:                 {args.flags_out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
