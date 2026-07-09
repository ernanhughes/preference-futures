"""Command-line interface for NewsEdits inspection and episode extraction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from preference_futures.newsedits.database import connect_read_only
from preference_futures.newsedits.extract import extract_from_database
from preference_futures.newsedits.models import ExtractionConfig
from preference_futures.newsedits.schema import discover_article_schema, sqlite_tables, table_columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-newsedits")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect SQLite tables and schema")
    inspect_parser.add_argument("--db", type=Path, required=True)
    inspect_parser.add_argument("--table")

    extract_parser = subparsers.add_parser("extract", help="extract canonical JSONL episodes")
    extract_parser.add_argument("--db", type=Path, required=True)
    extract_parser.add_argument("--table")
    extract_parser.add_argument("--out", type=Path, required=True)
    extract_parser.add_argument("--audit-out", type=Path, required=True)
    extract_parser.add_argument("--seed", type=int, default=0)
    extract_parser.add_argument("--max-articles", type=int, default=0)
    extract_parser.add_argument("--max-examples", type=int, default=0)
    extract_parser.add_argument("--sources", default="")
    extract_parser.add_argument("--context-before", type=int, default=1)
    extract_parser.add_argument("--context-after", type=int, default=1)
    extract_parser.add_argument("--min-sentence-chars", type=int, default=20)
    extract_parser.add_argument("--max-sentence-chars", type=int, default=500)
    extract_parser.add_argument("--min-edit-similarity", type=float, default=0.15)
    extract_parser.add_argument("--max-edit-similarity", type=float, default=0.98)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    connection = connect_read_only(args.db)
    try:
        if args.command == "inspect":
            payload = {
                "tables": {
                    table: table_columns(connection, table) for table in sqlite_tables(connection)
                },
                "detected_schema": asdict(discover_article_schema(connection, args.table)),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        schema = discover_article_schema(connection, args.table)
        config = ExtractionConfig(
            context_before=args.context_before,
            context_after=args.context_after,
            min_sentence_chars=args.min_sentence_chars,
            max_sentence_chars=args.max_sentence_chars,
            min_edit_similarity=args.min_edit_similarity,
            max_edit_similarity=args.max_edit_similarity,
        )
        sources = tuple(value.strip() for value in args.sources.split(",") if value.strip())
        result = extract_from_database(
            connection,
            schema,
            config=config,
            max_articles=args.max_articles,
            max_examples=args.max_examples,
            seed=args.seed,
            sources=sources,
        )
    finally:
        connection.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        for example in result.examples:
            stream.write(json.dumps(example.to_record(seed=args.seed), ensure_ascii=False) + "\n")

    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(result.audit.to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0
