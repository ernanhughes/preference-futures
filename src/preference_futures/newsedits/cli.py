"""Command-line interface for NewsEdits inspection and episode extraction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, TypeVar

from preference_futures.newsedits.database import connect_read_only, infer_source_name
from preference_futures.newsedits.extract import (
    extract_from_database,
    extract_from_split_database,
)
from preference_futures.newsedits.models import ExtractionConfig
from preference_futures.newsedits.schema import (
    discover_article_schema,
    discover_split_sentence_schema,
    sqlite_tables,
    table_columns,
    table_row_count,
)

SchemaT = TypeVar("SchemaT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-futures-newsedits")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect SQLite tables and schema")
    inspect_parser.add_argument("--db", type=Path, required=True)
    inspect_parser.add_argument("--table", help="preferred full article-version table")
    inspect_parser.add_argument("--split-table", help="preferred split-sentence table")

    extract_parser = subparsers.add_parser("extract", help="extract canonical JSONL episodes")
    extract_parser.add_argument("--db", type=Path, required=True)
    extract_parser.add_argument("--table", help="preferred full article-version table")
    extract_parser.add_argument("--split-table", help="preferred split-sentence table")
    extract_parser.add_argument(
        "--source-name",
        help="publisher name for source-specific split-sentence databases; inferred from filename",
    )
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
    input_format = "unknown"
    source_name: str | None = None
    try:
        if args.command == "inspect":
            article_schema, article_error = _try_discover(
                discover_article_schema,
                connection,
                args.table,
            )
            split_preferred = args.split_table or args.table
            split_schema, split_error = _try_discover(
                discover_split_sentence_schema,
                connection,
                split_preferred,
            )
            payload = {
                "database": str(args.db.expanduser().resolve()),
                "tables": {
                    table: {
                        "columns": table_columns(connection, table),
                        "rows": table_row_count(connection, table),
                    }
                    for table in sqlite_tables(connection)
                },
                "detected_schemas": {
                    "article_versions": None if article_schema is None else asdict(article_schema),
                    "split_sentences": None if split_schema is None else asdict(split_schema),
                },
                "schema_errors": {
                    "article_versions": article_error,
                    "split_sentences": split_error,
                },
                "inferred_source_name": infer_source_name(args.db),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if article_schema is not None or split_schema is not None else 2

        config = ExtractionConfig(
            context_before=args.context_before,
            context_after=args.context_after,
            min_sentence_chars=args.min_sentence_chars,
            max_sentence_chars=args.max_sentence_chars,
            min_edit_similarity=args.min_edit_similarity,
            max_edit_similarity=args.max_edit_similarity,
        )
        article_schema, article_error = _try_discover(
            discover_article_schema,
            connection,
            args.table,
        )
        if article_schema is not None:
            sources = tuple(value.strip() for value in args.sources.split(",") if value.strip())
            result = extract_from_database(
                connection,
                article_schema,
                config=config,
                max_articles=args.max_articles,
                max_examples=args.max_examples,
                seed=args.seed,
                sources=sources,
            )
            input_format = "article_versions"
        else:
            split_preferred = args.split_table or args.table
            split_schema, split_error = _try_discover(
                discover_split_sentence_schema,
                connection,
                split_preferred,
            )
            if split_schema is None:
                raise ValueError(
                    "no supported NewsEdits schema was detected. "
                    f"Article-version attempt: {article_error}. "
                    f"Split-sentence attempt: {split_error}."
                )
            source_name = infer_source_name(args.db, args.source_name)
            result = extract_from_split_database(
                connection,
                split_schema,
                source_name=source_name,
                config=config,
                max_articles=args.max_articles,
                max_examples=args.max_examples,
                seed=args.seed,
            )
            input_format = "split_sentences"
    finally:
        connection.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        for example in result.examples:
            stream.write(json.dumps(example.to_record(seed=args.seed), ensure_ascii=False) + "\n")

    audit_record = {
        **result.audit.to_record(),
        "input_format": input_format,
        "source_name": source_name,
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _try_discover(discoverer, connection, preferred_table) -> tuple[SchemaT | None, str | None]:
    try:
        return discoverer(connection, preferred_table), None
    except ValueError as exc:
        return None, str(exc)
