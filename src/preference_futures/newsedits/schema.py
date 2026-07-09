"""SQLite schema discovery for NewsEdits releases and compatible fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from preference_futures.newsedits.models import ArticleSchema

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source": ("source", "publisher", "outlet"),
    "article_id": ("a_id", "article_id", "articleid", "story_id"),
    "version_id": ("version_id", "v_id", "version", "revision_id"),
    "text": ("text", "article_text", "body", "content"),
    "created": ("created", "created_at", "timestamp", "published_at", "date"),
    "title": ("title", "headline"),
    "num_versions": ("num_versions", "version_count", "n_versions"),
}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sqlite_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [str(row[1]) for row in rows]


def resolve_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def discover_article_schema(
    connection: sqlite3.Connection,
    preferred_table: str | None = None,
) -> ArticleSchema:
    tables = sqlite_tables(connection)
    if not tables:
        raise ValueError("the SQLite database contains no tables")

    candidates = [preferred_table] if preferred_table is not None else tables
    diagnostics: list[str] = []
    for table in candidates:
        if table not in tables:
            diagnostics.append(f"{table}: table not found")
            continue
        columns = table_columns(connection, table)
        resolved = {
            logical: resolve_column(columns, aliases)
            for logical, aliases in COLUMN_ALIASES.items()
        }
        required = ("source", "article_id", "version_id", "text")
        missing = [name for name in required if resolved[name] is None]
        if missing:
            diagnostics.append(f"{table}: missing {', '.join(missing)}")
            continue
        return ArticleSchema(
            table=table,
            source=str(resolved["source"]),
            article_id=str(resolved["article_id"]),
            version_id=str(resolved["version_id"]),
            text=str(resolved["text"]),
            created=resolved["created"],
            title=resolved["title"],
            num_versions=resolved["num_versions"],
        )

    detail = "; ".join(diagnostics)
    raise ValueError(
        "could not find an article-version table with source, article ID, "
        f"version ID and text columns: {detail}"
    )
