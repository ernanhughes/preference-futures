"""Command-line interface for episode context viability auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from preference_futures.audit.report import (
    build_context_viability_report,
    load_episode_records,
    render_context_viability_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-audit")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes_path = args.episodes.expanduser().resolve()
    records = load_episode_records(episodes_path)
    report = build_context_viability_report(records, source_path=episodes_path)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(
        render_context_viability_markdown(report),
        encoding="utf-8",
    )

    target = report["target_balance"]
    context = report["context"]
    pairs = report["pair_reuse"]
    print("Context viability audit complete.")
    print(f"  Episodes:          {report['dataset']['episodes']}")
    print(f"  Lineages:          {report['dataset']['lineages']}")
    print(f"  Future revised:    {target['future_revised_rate']:.4f}")
    print(f"  Boundary artifacts:{context['source_boundary_artifact_rate']:.4f}")
    print(f"  Reversal episodes: {pairs['reversal_episode_rate']:.4f}")
    print(f"  JSON:              {args.json_out.expanduser().resolve()}")
    print(f"  Markdown:          {args.markdown_out.expanduser().resolve()}")
    return 0 if all(report["gates"].values()) else 2
