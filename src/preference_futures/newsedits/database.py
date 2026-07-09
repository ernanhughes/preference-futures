"""SQLite access for NewsEdits article-version and split-sentence data."""

from __future__ import annotations

import random
import re
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from preference_futures.newsedits.models import (
    ArticleSchema,
    ArticleVersion,
    SplitSentenceSchema,
)
from preference_futures.newsedits.schema import quote_identifier
from preference_futures.newsedits.text import normalise_space


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite database without permitting accidental mutation."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def infer_source_name(path: Path, explicit_source: str | None = None) -> str:
    """Infer the publisher from official names such as ``nyt-matched-sentences.db``."""

    if explicit_source is not None and explicit_source.strip():
        return explicit_source.strip()

    name = path.name
    for suffix in (".db.gz", ".sqlite.gz", ".sqlite3.gz", ".db", ".sqlite", ".sqlite3"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(
        r"[-_](matched[-_]?sentences|sentence[-_]?diffs|processed)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return name or "unknown"


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
    return _reservoir_sample(connection.execute(sql, params), max_items=max_articles, seed=seed)


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


def sample_split_article_ids(
    connection: sqlite3.Connection,
    schema: SplitSentenceSchema,
    *,
    max_articles: int = 0,
    seed: int = 0,
) -> list[tuple[Any, int]]:
    """Sample official NewsEdits entries containing at least three versions."""

    table = quote_identifier(schema.table)
    article_col = quote_identifier(schema.article_id)
    version_col = quote_identifier(schema.version_id)
    sentence_col = quote_identifier(schema.sentence)
    sql = f"""
        SELECT {article_col}, COUNT(DISTINCT {version_col})
        FROM {table}
        WHERE {sentence_col} IS NOT NULL
          AND LENGTH(TRIM({sentence_col})) > 0
        GROUP BY {article_col}
        HAVING COUNT(DISTINCT {version_col}) >= 3
    """
    return _reservoir_sample(connection.execute(sql), max_items=max_articles, seed=seed)


def iter_split_article_versions(
    connection: sqlite3.Connection,
    schema: SplitSentenceSchema,
    selected_ids: Sequence[tuple[Any, int]],
    *,
    source_name: str,
) -> Iterator[tuple[tuple[str, str], tuple[ArticleVersion, ...]]]:
    """Reconstruct complete versions from official ``split_sentences`` rows."""

    if not selected_ids:
        return
    connection.execute("DROP TABLE IF EXISTS temp.pf_selected_entries")
    connection.execute(
        "CREATE TEMP TABLE pf_selected_entries "
        "(article_value, version_count INTEGER)"
    )
    connection.executemany(
        "INSERT INTO pf_selected_entries VALUES (?, ?)",
        selected_ids,
    )

    table = quote_identifier(schema.table)
    article_col = quote_identifier(schema.article_id)
    version_col = quote_identifier(schema.version_id)
    sentence_id_col = quote_identifier(schema.sentence_id)
    sentence_col = quote_identifier(schema.sentence)
    sql = f"""
        SELECT
            s.{article_col}, s.{version_col}, s.{sentence_id_col}, s.{sentence_col}
        FROM {table} AS s
        INNER JOIN temp.pf_selected_entries AS k
          ON s.{article_col} = k.article_value
        WHERE s.{sentence_col} IS NOT NULL
          AND LENGTH(TRIM(s.{sentence_col})) > 0
        ORDER BY
            s.{article_col},
            CAST(s.{version_col} AS REAL),
            s.{version_col},
            CAST(s.{sentence_id_col} AS INTEGER),
            s.{sentence_id_col}
    """

    current_article: str | None = None
    current_version: str | None = None
    current_sentences: list[str] = []
    versions: list[ArticleVersion] = []

    def flush_version() -> None:
        nonlocal current_sentences
        if current_article is None or current_version is None or not current_sentences:
            return
        versions.append(
            ArticleVersion(
                source=source_name,
                article_id=current_article,
                version_id=current_version,
                text=" ".join(current_sentences),
            )
        )
        current_sentences = []

    for article_value, version_value, _sentence_index, sentence_value in connection.execute(sql):
        article = str(article_value)
        version = str(version_value)
        if current_article is None:
            current_article = article
            current_version = version
        elif article != current_article:
            flush_version()
            yield (source_name, current_article), tuple(versions)
            current_article = article
            current_version = version
            versions = []
            current_sentences = []
        elif version != current_version:
            flush_version()
            current_version = version

        sentence = normalise_space(sentence_value)
        if sentence:
            current_sentences.append(sentence)

    if current_article is not None:
        flush_version()
        if versions:
            yield (source_name, current_article), tuple(versions)


def _reservoir_sample(
    rows: Iterator[Sequence[Any]],
    *,
    max_items: int,
    seed: int,
) -> list[tuple[Any, ...]]:
    rng = random.Random(seed)
    reservoir: list[tuple[Any, ...]] = []
    for seen, row in enumerate(rows, start=1):
        item = tuple(row)
        if max_items <= 0 or len(reservoir) < max_items:
            reservoir.append(item)
            continue
        replacement = rng.randrange(seen)
        if replacement < max_items:
            reservoir[replacement] = item
    return reservoir


def _version_sort_key(version: ArticleVersion) -> tuple[object, ...]:
    created_missing = version.created is None
    created = version.created or ""
    natural_version = tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", version.version_id)
        if part
    )
    return (created_missing, created, natural_version)
