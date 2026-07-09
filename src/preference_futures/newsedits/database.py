"""SQLite access for NewsEdits article-version data."""

from __future__ import annotations

import random
import re
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from preference_futures.newsedits.models import ArticleSchema, ArticleVersion
from preference_futures.newsedits.schema import quote_identifier


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite database without permitting accidental mutation."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def sample_article_keys(
    connection: sqlite3.Connection,
    schema: ArticleSchema,
    *,
    max_articles: int = 0,
    seed: int = 0,
    sources: Sequence[str] = (),
) -> list[tuple[Any, Any, int]]:
    """Return deterministic reservoir-sampled article keys with at least three versions."""

    table = quote_identifier(schema.table)
    source_col = quote_identifier(schema.source)
    article_col = quote_identifier(schema.article_id)
    text_col = quote_identifier(schema.text)
    conditions = [f"{text_col} IS NOT NULL", f"LENGTH(TRIM({text_col})) > 0"]
    params: list[Any] = []
    if sources:
        placeholders = ",".join("?" for _ in sources)
        conditions.append(f"CAST({source_col} AS TEXT) IN ({placeholders})")
        params.extend(sources)

    sql = f"""
        SELECT {source_col}, {article_col}, COUNT(*)
        FROM {table}
        WHERE {' AND '.join(conditions)}
        GROUP BY {source_col}, {article_col}
        HAVING COUNT(*) >= 3
    """
    rng = random.Random(seed)
    reservoir: list[tuple[Any, Any, int]] = []
    for seen, row in enumerate(connection.execute(sql, params), start=1):
        item = (row[0], row[1], int(row[2]))
        if max_articles <= 0 or len(reservoir) < max_articles:
            reservoir.append(item)
            continue
        replacement = rng.randrange(seen)
        if replacement < max_articles:
            reservoir[replacement] = item
    return reservoir


def iter_article_versions(
    connection: sqlite3.Connection,
    schema: ArticleSchema,
    selected_keys: Sequence[tuple[Any, Any, int]],
) -> Iterator[tuple[tuple[str, str], tuple[ArticleVersion, ...]]]:
    """Yield selected article lineages with versions in deterministic temporal order."""

    if not selected_keys:
        return
    connection.execute("DROP TABLE IF EXISTS temp.pf_selected_articles")
    connection.execute(
        "CREATE TEMP TABLE pf_selected_articles "
        "(source_value, article_value, version_count INTEGER)"
    )
    connection.executemany(
        "INSERT INTO pf_selected_articles VALUES (?, ?, ?)",
        selected_keys,
    )

    table = quote_identifier(schema.table)
    source_col = quote_identifier(schema.source)
    article_col = quote_identifier(schema.article_id)
    version_col = quote_identifier(schema.version_id)
    text_col = quote_identifier(schema.text)
    created_expr = (
        f"a.{quote_identifier(schema.created)}" if schema.created is not None else "NULL"
    )
    title_expr = f"a.{quote_identifier(schema.title)}" if schema.title is not None else "NULL"
    sql = f"""
        SELECT
            a.{source_col}, a.{article_col}, a.{version_col}, a.{text_col},
            {created_expr}, {title_expr}
        FROM {table} AS a
        INNER JOIN temp.pf_selected_articles AS k
          ON a.{source_col} = k.source_value
         AND a.{article_col} = k.article_value
        WHERE a.{text_col} IS NOT NULL
          AND LENGTH(TRIM(a.{text_col})) > 0
        ORDER BY a.{source_col}, a.{article_col}
    """

    current_key: tuple[str, str] | None = None
    current_versions: list[ArticleVersion] = []
    for row in connection.execute(sql):
        key = (str(row[0]), str(row[1]))
        if current_key is not None and key != current_key:
            yield current_key, tuple(sorted(current_versions, key=_version_sort_key))
            current_versions = []
        current_key = key
        current_versions.append(
            ArticleVersion(
                source=key[0],
                article_id=key[1],
                version_id=str(row[2]),
                text=str(row[3]),
                created=None if row[4] is None else str(row[4]),
                title="" if row[5] is None else str(row[5]),
            )
        )
    if current_key is not None and current_versions:
        yield current_key, tuple(sorted(current_versions, key=_version_sort_key))


def _version_sort_key(version: ArticleVersion) -> tuple[object, ...]:
    created_missing = version.created is None
    created = version.created or ""
    natural_version = tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", version.version_id)
        if part
    )
    return (created_missing, created, natural_version)
