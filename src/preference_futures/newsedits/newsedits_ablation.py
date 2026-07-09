#!/usr/bin/env python3
r"""
PreferenceFutures — NewsEdits mechanism-ablation probe.

Research question
-----------------

Does a real human revision event contain incremental information about the
future of the revised sentence?

For three consecutive article versions V0 -> V1 -> V2:

    rejected sentence: sentence in V0
    retained sentence: its one-to-one replacement in V1
    future: whether the retained V1 sentence is revised or removed in V2

The baseline already sees the retained sentence and its V1 context:

    P(F | context, retained sentence, metadata)

The preference-informed model additionally sees what the journalist replaced:

    P(F | context, retained sentence, rejected sentence, edit evidence, metadata)

Preference Future Information:

    PFI = Loss(baseline) - Loss(preference-informed)

Positive held-out PFI means the linked preference bundle carries information
about the next revision beyond the retained sentence itself. This v4 probe
separates rejected-text semantics, edit geometry and lexical relationship.

This is a revealed-revision-preference experiment, not an explicit A/B-vote
experiment and not a causal estimate.

Expected NewsEdits source
-------------------------

The official NewsEdits download provides source-specific compressed SQLite
databases such as ``nyt-matched-sentences.db.gz``. After decompression, the
database normally contains ``split_sentences`` and ``matched_sentences``.
This script reads ``split_sentences`` directly and reconstructs complete
article versions from ``entry_id``, ``version``, ``sent_idx`` and ``sentence``.

A full-article table with SOURCE, A_ID, VERSION_ID and TEXT is also supported
for compatible exports.

Dependencies
------------

    pip install pandas numpy scikit-learn

Optional, only for Parquet episode caches:

    pip install pyarrow

Examples
--------

Inspect the database schema:

    python preference_futures_newsedits_v4_ablation.py \
      --db /path/to/newsedits.db \
      --inspect-only

Windows PowerShell smoke test:

    python preference_futures_newsedits_v4_ablation.py `
      --db C:\data\newsedits.db `
      --max-articles 5000 `
      --max-episodes 50000 `
      --seeds 1,2,3 `
      --bootstrap-samples 500 `
      --episode-cache newsedits_smoke_episodes.csv.gz `
      --ablation-profile core `
      --out newsedits_ablation_smoke_runs.csv `
      --summary-out newsedits_ablation_smoke_summary.csv

Larger run:

    python preference_futures_newsedits_v4_ablation.py `
      --db C:\data\newsedits.db `
      --max-articles 100000 `
      --seeds 1,2,3,4,5,6,7,8,9,10 `
      --bootstrap-samples 5000 `
      --episode-cache newsedits_full_episodes.csv.gz `
      --ablation-profile full `
      --out newsedits_ablation_full_runs.csv `
      --summary-out newsedits_ablation_full_summary.csv

Use --max-articles 0 to request every qualifying article. Build and validate a
smaller cache first: the complete corpus is very large.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import math
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArticleSchema:
    table: str
    source: str
    article_id: str
    version_id: str
    text: str
    created: str | None
    title: str | None
    num_versions: str | None


@dataclasses.dataclass(frozen=True)
class SplitSentenceSchema:
    table: str
    article_id: str
    version_id: str
    sentence_id: str
    sentence: str


@dataclasses.dataclass
class ResultRow:
    track: str
    condition: str
    seed: int
    target: str
    feature_set: str
    n_train: int
    n_test: int
    n_train_groups: int
    n_test_groups: int
    loss: float
    brier: float
    auc: float
    average_precision: float
    accuracy: float
    train_prevalence: float
    test_prevalence: float
    mean_predicted_probability: float
    probability_min: float
    probability_p01: float
    probability_p05: float
    probability_median: float
    probability_p95: float
    probability_p99: float
    probability_max: float
    calibration_gap: float
    solver: str
    converged: bool
    n_iter: int
    null_log_loss: float
    null_brier: float


@dataclasses.dataclass
class SummaryRow:
    track: str
    condition: str
    comparison: str
    metric: str
    reference_feature_set: str
    candidate_feature_set: str
    n_seeds: int
    mean_gain: float
    seed_std: float
    ci_low: float
    ci_high: float
    positive_seeds: int
    confidence_level: float
    bootstrap_samples: int


@dataclasses.dataclass
class EvaluationBundle:
    rows: list[ResultRow]
    group_loss_stats: pd.DataFrame


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_int_list(value: str | None, fallback: int) -> list[int]:
    if value is None or not value.strip():
        return [fallback]
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("--seeds did not contain any integers.")
    return values


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def normalise_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_sentence(value: str) -> str:
    text = normalise_space(value).lower()
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text


def tokenise(value: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", value.lower())


def lexical_jaccard(left: str, right: str) -> float:
    a = set(tokenise(left))
    b = set(tokenise(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def stable_int_hash(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def sentence_split(text: str) -> list[str]:
    """Dependency-free sentence splitter suitable for a first corpus probe."""
    cleaned = normalise_space(text)
    if not cleaned:
        return []

    # Split after likely sentence punctuation, or at paragraph/newline boundaries.
    pieces = re.split(
        r"(?<=[.!?])\s+(?=(?:[\"'“‘(\[]?[A-Z0-9]))|(?:\s*\n+\s*)",
        cleaned,
    )
    sentences = [normalise_space(piece) for piece in pieces]
    return [sentence for sentence in sentences if sentence]


def valid_sentence(
    sentence: str,
    *,
    min_chars: int,
    max_chars: int,
) -> bool:
    length = len(sentence)
    if length < min_chars or length > max_chars:
        return False
    return len(tokenise(sentence)) >= 3


def log_loss_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
    y = y_true.astype(float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.square(probabilities.astype(float) - y_true.astype(float))


# ---------------------------------------------------------------------------
# SQLite schema discovery
# ---------------------------------------------------------------------------


COLUMN_ALIASES = {
    "source": ["source", "publisher", "outlet"],
    "article_id": ["a_id", "article_id", "articleid", "story_id"],
    "version_id": ["version_id", "v_id", "version", "revision_id"],
    "text": ["text", "article_text", "body", "content"],
    "created": ["created", "created_at", "timestamp", "published_at", "date"],
    "title": ["title", "headline"],
    "num_versions": ["num_versions", "version_count", "n_versions"],
}


SPLIT_COLUMN_ALIASES = {
    "article_id": ["entry_id", "a_id", "article_id", "articleid", "story_id"],
    "version_id": ["version", "version_id", "v_id", "revision_id"],
    "sentence_id": ["sent_idx", "sentence_id", "sent_id", "sentence_index"],
    "sentence": ["sentence", "sent", "text"],
}


def sqlite_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> list[str]:
    rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    return [str(row[1]) for row in rows]


def resolve_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def discover_article_schema(
    connection: sqlite3.Connection,
    preferred_table: str | None,
) -> ArticleSchema:
    tables = sqlite_tables(connection)
    if not tables:
        raise ValueError("The SQLite database contains no tables.")

    candidates = [preferred_table] if preferred_table else tables
    candidates = [table for table in candidates if table is not None]

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

        required = ["source", "article_id", "version_id", "text"]
        missing = [name for name in required if resolved[name] is None]
        diagnostics.append(
            f"{table}: columns={columns}; missing_required={missing}"
        )
        if missing:
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

    detail = "\n".join(f"  - {line}" for line in diagnostics)
    raise ValueError(
        "Could not find an article-version table with source, article ID, "
        f"version ID and text columns.\n{detail}"
    )


def discover_split_sentence_schema(
    connection: sqlite3.Connection,
    preferred_table: str | None = None,
) -> SplitSentenceSchema:
    """Discover the official NewsEdits ``split_sentences`` table schema."""
    tables = sqlite_tables(connection)
    if not tables:
        raise ValueError("The SQLite database contains no tables.")

    if preferred_table:
        candidates = [preferred_table]
    else:
        candidates = [
            table
            for table in tables
            if table.lower() == "split_sentences"
        ] + [
            table
            for table in tables
            if table.lower() != "split_sentences"
        ]

    diagnostics: list[str] = []
    for table in candidates:
        if table not in tables:
            diagnostics.append(f"{table}: table not found")
            continue

        columns = table_columns(connection, table)
        resolved = {
            logical: resolve_column(columns, aliases)
            for logical, aliases in SPLIT_COLUMN_ALIASES.items()
        }
        missing = [
            logical for logical, value in resolved.items()
            if value is None
        ]
        diagnostics.append(
            f"{table}: columns={columns}; missing_required={missing}"
        )
        if missing:
            continue

        return SplitSentenceSchema(
            table=table,
            article_id=str(resolved["article_id"]),
            version_id=str(resolved["version_id"]),
            sentence_id=str(resolved["sentence_id"]),
            sentence=str(resolved["sentence"]),
        )

    detail = "\n".join(f"  - {line}" for line in diagnostics)
    raise ValueError(
        "Could not find an official NewsEdits split-sentence table with "
        "entry/article ID, version, sentence index and sentence text.\n"
        f"{detail}"
    )


def inspect_database(
    connection: sqlite3.Connection,
    preferred_table: str | None,
    preferred_split_table: str | None = None,
) -> None:
    print_header("SQLite database inspection")
    for table in sqlite_tables(connection):
        columns = table_columns(connection, table)
        try:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(table)}"
            ).fetchone()[0]
        except sqlite3.DatabaseError:
            count = "unavailable"
        print(f"{table}: rows={count:,}" if isinstance(count, int) else f"{table}: rows={count}")
        print("  " + ", ".join(columns))

    article_error: str | None = None
    split_error: str | None = None

    try:
        schema = discover_article_schema(connection, preferred_table)
        print_header("Detected full-article schema")
        print(dataclasses.asdict(schema))
    except ValueError as exc:
        article_error = str(exc)

    try:
        split_schema = discover_split_sentence_schema(
            connection,
            preferred_split_table,
        )
        print_header("Detected official split-sentence schema")
        print(dataclasses.asdict(split_schema))
    except ValueError as exc:
        split_error = str(exc)

    if article_error and split_error:
        raise ValueError(
            "No supported NewsEdits schema was detected.\n\n"
            f"Full-article attempt:\n{article_error}\n\n"
            f"Split-sentence attempt:\n{split_error}"
        )


# ---------------------------------------------------------------------------
# Article sampling and version loading
# ---------------------------------------------------------------------------


def reservoir_sample_article_keys(
    connection: sqlite3.Connection,
    schema: ArticleSchema,
    *,
    max_articles: int,
    seed: int,
    sources: Sequence[str],
) -> list[tuple[Any, Any, int]]:
    table = quote_identifier(schema.table)
    source_col = quote_identifier(schema.source)
    article_col = quote_identifier(schema.article_id)
    text_col = quote_identifier(schema.text)

    conditions = [
        f"{text_col} IS NOT NULL",
        f"LENGTH(TRIM({text_col})) > 0",
    ]
    params: list[Any] = []

    if sources:
        placeholders = ",".join("?" for _ in sources)
        conditions.append(f"CAST({source_col} AS TEXT) IN ({placeholders})")
        params.extend(sources)

    sql = f"""
        SELECT
            {source_col} AS source_value,
            {article_col} AS article_value,
            COUNT(*) AS version_count
        FROM {table}
        WHERE {' AND '.join(conditions)}
        GROUP BY {source_col}, {article_col}
        HAVING COUNT(*) >= 3
    """

    rng = random.Random(seed)
    reservoir: list[tuple[Any, Any, int]] = []
    seen = 0

    cursor = connection.execute(sql, params)
    for source_value, article_value, version_count in cursor:
        item = (source_value, article_value, int(version_count))
        seen += 1

        if max_articles <= 0:
            reservoir.append(item)
            continue

        if len(reservoir) < max_articles:
            reservoir.append(item)
        else:
            replacement = rng.randrange(seen)
            if replacement < max_articles:
                reservoir[replacement] = item

    print(
        f"Qualifying articles with 3+ versions: {seen:,}; "
        f"selected: {len(reservoir):,}"
    )
    return reservoir


def load_selected_versions(
    connection: sqlite3.Connection,
    schema: ArticleSchema,
    selected_keys: Sequence[tuple[Any, Any, int]],
) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
    if not selected_keys:
        return

    connection.execute("DROP TABLE IF EXISTS temp.pf_selected_articles")
    connection.execute(
        """
        CREATE TEMP TABLE pf_selected_articles (
            source_value,
            article_value,
            version_count INTEGER
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO pf_selected_articles
            (source_value, article_value, version_count)
        VALUES (?, ?, ?)
        """,
        selected_keys,
    )

    table = quote_identifier(schema.table)
    source_col = quote_identifier(schema.source)
    article_col = quote_identifier(schema.article_id)
    version_col = quote_identifier(schema.version_id)
    text_col = quote_identifier(schema.text)

    created_expr = (
        f"a.{quote_identifier(schema.created)}"
        if schema.created
        else "NULL"
    )
    title_expr = (
        f"a.{quote_identifier(schema.title)}"
        if schema.title
        else "NULL"
    )

    sql = f"""
        SELECT
            a.{source_col} AS source_value,
            a.{article_col} AS article_value,
            a.{version_col} AS version_value,
            a.{text_col} AS text_value,
            {created_expr} AS created_value,
            {title_expr} AS title_value,
            k.version_count AS sampled_version_count
        FROM {table} AS a
        INNER JOIN temp.pf_selected_articles AS k
          ON a.{source_col} = k.source_value
         AND a.{article_col} = k.article_value
        WHERE a.{text_col} IS NOT NULL
          AND LENGTH(TRIM(a.{text_col})) > 0
        ORDER BY
            a.{source_col},
            a.{article_col},
            CASE WHEN created_value IS NULL THEN 1 ELSE 0 END,
            created_value,
            a.{version_col}
    """

    current_key: tuple[str, str] | None = None
    current_versions: list[dict[str, Any]] = []

    for row in connection.execute(sql):
        key = (str(row[0]), str(row[1]))
        version = {
            "source": str(row[0]),
            "article_id": str(row[1]),
            "version_id": str(row[2]),
            "text": str(row[3]),
            "created": None if row[4] is None else str(row[4]),
            "title": "" if row[5] is None else str(row[5]),
            "n_versions": int(row[6]),
        }

        if current_key is None:
            current_key = key

        if key != current_key:
            yield current_key, current_versions
            current_key = key
            current_versions = []

        current_versions.append(version)

    if current_key is not None and current_versions:
        yield current_key, current_versions



# ---------------------------------------------------------------------------
# Official NewsEdits split-sentence loading
# ---------------------------------------------------------------------------


def infer_source_name(db_path: Path, explicit_source: str | None) -> str:
    if explicit_source:
        return explicit_source

    name = db_path.name
    for suffix in [".db.gz", ".sqlite.gz", ".sqlite3.gz", ".db", ".sqlite", ".sqlite3"]:
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


def reservoir_sample_split_article_ids(
    connection: sqlite3.Connection,
    schema: SplitSentenceSchema,
    *,
    max_articles: int,
    seed: int,
) -> list[tuple[Any, int]]:
    table = quote_identifier(schema.table)
    article_col = quote_identifier(schema.article_id)
    version_col = quote_identifier(schema.version_id)
    sentence_col = quote_identifier(schema.sentence)

    sql = f"""
        SELECT
            {article_col} AS article_value,
            COUNT(DISTINCT {version_col}) AS version_count
        FROM {table}
        WHERE {sentence_col} IS NOT NULL
          AND LENGTH(TRIM({sentence_col})) > 0
        GROUP BY {article_col}
        HAVING COUNT(DISTINCT {version_col}) >= 3
    """

    rng = random.Random(seed)
    reservoir: list[tuple[Any, int]] = []
    seen = 0

    for article_value, version_count in connection.execute(sql):
        item = (article_value, int(version_count))
        seen += 1

        if max_articles <= 0:
            reservoir.append(item)
        elif len(reservoir) < max_articles:
            reservoir.append(item)
        else:
            replacement = rng.randrange(seen)
            if replacement < max_articles:
                reservoir[replacement] = item

    print(
        f"Qualifying entries with 3+ versions: {seen:,}; "
        f"selected: {len(reservoir):,}"
    )
    return reservoir


def load_selected_split_versions(
    connection: sqlite3.Connection,
    schema: SplitSentenceSchema,
    selected_ids: Sequence[tuple[Any, int]],
    *,
    source_name: str,
) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
    if not selected_ids:
        return

    connection.execute("DROP TABLE IF EXISTS temp.pf_selected_entries")
    connection.execute(
        """
        CREATE TEMP TABLE pf_selected_entries (
            article_value,
            version_count INTEGER
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO pf_selected_entries
            (article_value, version_count)
        VALUES (?, ?)
        """,
        selected_ids,
    )

    table = quote_identifier(schema.table)
    article_col = quote_identifier(schema.article_id)
    version_col = quote_identifier(schema.version_id)
    sentence_id_col = quote_identifier(schema.sentence_id)
    sentence_col = quote_identifier(schema.sentence)

    sql = f"""
        SELECT
            s.{article_col} AS article_value,
            s.{version_col} AS version_value,
            s.{sentence_id_col} AS sentence_index,
            s.{sentence_col} AS sentence_value,
            k.version_count AS sampled_version_count
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
    versions: list[dict[str, Any]] = []
    sampled_version_count = 0

    def flush_version() -> None:
        nonlocal current_version, current_sentences, versions
        if current_version is None or not current_sentences:
            return
        versions.append(
            {
                "source": source_name,
                "article_id": str(current_article),
                "version_id": str(current_version),
                "text": " ".join(current_sentences),
                "created": None,
                "title": "",
                "n_versions": sampled_version_count,
            }
        )
        current_sentences = []

    for (
        article_value,
        version_value,
        _sentence_index,
        sentence_value,
        version_count,
    ) in connection.execute(sql):
        article_value_str = str(article_value)
        version_value_str = str(version_value)

        if current_article is None:
            current_article = article_value_str
            current_version = version_value_str
            sampled_version_count = int(version_count)

        if article_value_str != current_article:
            flush_version()
            yield (source_name, current_article), versions
            current_article = article_value_str
            current_version = version_value_str
            current_sentences = []
            versions = []
            sampled_version_count = int(version_count)
        elif version_value_str != current_version:
            flush_version()
            current_version = version_value_str

        sentence = normalise_space(sentence_value)
        if sentence:
            current_sentences.append(sentence)

    if current_article is not None:
        flush_version()
        if versions:
            yield (source_name, current_article), versions


def build_episode_dataframe_from_split_sentences(
    connection: sqlite3.Connection,
    schema: SplitSentenceSchema,
    *,
    source_name: str,
    max_articles: int,
    max_episodes: int,
    sampling_seed: int,
    context_before: int,
    context_after: int,
    min_sentence_chars: int,
    max_sentence_chars: int,
    min_edit_similarity: float,
    max_edit_similarity: float,
) -> pd.DataFrame:
    selected_ids = reservoir_sample_split_article_ids(
        connection,
        schema,
        max_articles=max_articles,
        seed=sampling_seed,
    )

    records: list[dict[str, Any]] = []
    processed_articles = 0

    for key, versions in load_selected_split_versions(
        connection,
        schema,
        selected_ids,
        source_name=source_name,
    ):
        processed_articles += 1
        records.extend(
            extract_episodes_from_article(
                key,
                versions,
                context_before=context_before,
                context_after=context_after,
                min_sentence_chars=min_sentence_chars,
                max_sentence_chars=max_sentence_chars,
                min_edit_similarity=min_edit_similarity,
                max_edit_similarity=max_edit_similarity,
            )
        )

        if processed_articles % 1000 == 0:
            print(
                f"Processed articles={processed_articles:,}; "
                f"episodes={len(records):,}"
            )

        if max_episodes > 0 and len(records) >= max_episodes:
            records = records[:max_episodes]
            break

    if not records:
        raise ValueError(
            "No revision-lineage episodes were extracted from split_sentences. "
            "Try more articles or relax the edit-similarity filters."
        )

    frame = pd.DataFrame.from_records(records)
    return frame.drop_duplicates(subset=["episode_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Revision lineage extraction
# ---------------------------------------------------------------------------


def sentence_fate_map(
    middle_sentences: Sequence[str],
    future_sentences: Sequence[str],
) -> dict[int, int]:
    """Return 0=unchanged in next version, 1=revised/removed."""
    middle_norm = [normalise_sentence(value) for value in middle_sentences]
    future_norm = [normalise_sentence(value) for value in future_sentences]

    matcher = difflib.SequenceMatcher(
        a=middle_norm,
        b=future_norm,
        autojunk=False,
    )

    fate: dict[int, int] = {}
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            for index in range(i1, i2):
                fate[index] = 0
        elif tag in {"replace", "delete"}:
            for index in range(i1, i2):
                fate[index] = 1
        # Insertions add future material but do not change an existing V1 sentence.
    return fate


def extract_episodes_from_article(
    key: tuple[str, str],
    versions: Sequence[dict[str, Any]],
    *,
    context_before: int,
    context_after: int,
    min_sentence_chars: int,
    max_sentence_chars: int,
    min_edit_similarity: float,
    max_edit_similarity: float,
) -> list[dict[str, Any]]:
    if len(versions) < 3:
        return []

    split_versions = [sentence_split(version["text"]) for version in versions]
    article_key = f"{key[0]}::{key[1]}"
    episodes: list[dict[str, Any]] = []

    for version_index in range(len(versions) - 2):
        old_version = versions[version_index]
        middle_version = versions[version_index + 1]
        future_version = versions[version_index + 2]

        old_sentences = split_versions[version_index]
        middle_sentences = split_versions[version_index + 1]
        future_sentences = split_versions[version_index + 2]

        if not old_sentences or not middle_sentences or not future_sentences:
            continue

        old_norm = [normalise_sentence(value) for value in old_sentences]
        middle_norm = [normalise_sentence(value) for value in middle_sentences]

        current_matcher = difflib.SequenceMatcher(
            a=old_norm,
            b=middle_norm,
            autojunk=False,
        )
        future_fate = sentence_fate_map(middle_sentences, future_sentences)

        for tag, i1, i2, j1, j2 in current_matcher.get_opcodes():
            # Start with clean one-to-one replacements. This gives a defensible
            # rejected/retained pair without ambiguous split/merge attribution.
            if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) != 1:
                continue

            rejected = old_sentences[i1]
            retained = middle_sentences[j1]

            if not valid_sentence(
                rejected,
                min_chars=min_sentence_chars,
                max_chars=max_sentence_chars,
            ):
                continue
            if not valid_sentence(
                retained,
                min_chars=min_sentence_chars,
                max_chars=max_sentence_chars,
            ):
                continue
            if j1 not in future_fate:
                continue

            similarity = difflib.SequenceMatcher(
                a=normalise_sentence(rejected),
                b=normalise_sentence(retained),
                autojunk=False,
            ).ratio()
            if similarity < min_edit_similarity:
                continue
            if similarity > max_edit_similarity:
                continue

            before_start = max(0, j1 - context_before)
            after_end = min(len(middle_sentences), j1 + 1 + context_after)
            preceding = " ".join(middle_sentences[before_start:j1])
            following = " ".join(middle_sentences[j1 + 1:after_end])

            rejected_tokens = len(tokenise(rejected))
            retained_tokens = len(tokenise(retained))

            episodes.append(
                {
                    "episode_id": (
                        f"{article_key}::{old_version['version_id']}"
                        f"->{middle_version['version_id']}::{j1}"
                    ),
                    "article_key": article_key,
                    "source": middle_version["source"],
                    "article_id": middle_version["article_id"],
                    "title": middle_version["title"],
                    "old_version_id": old_version["version_id"],
                    "retained_version_id": middle_version["version_id"],
                    "future_version_id": future_version["version_id"],
                    "version_index": version_index + 1,
                    "n_versions": len(versions),
                    "sentence_position": (
                        j1 / max(1, len(middle_sentences) - 1)
                    ),
                    "context_before": preceding,
                    "retained_sentence": retained,
                    "context_after": following,
                    "rejected_sentence": rejected,
                    "retained_chars": len(retained),
                    "rejected_chars": len(rejected),
                    "retained_tokens": retained_tokens,
                    "rejected_tokens": rejected_tokens,
                    "char_delta": len(retained) - len(rejected),
                    "token_delta": retained_tokens - rejected_tokens,
                    "edit_similarity": similarity,
                    "lexical_jaccard": lexical_jaccard(rejected, retained),
                    "revised_again_next_version": int(future_fate[j1]),
                }
            )

    return episodes


def build_episode_dataframe(
    connection: sqlite3.Connection,
    schema: ArticleSchema,
    *,
    max_articles: int,
    max_episodes: int,
    sampling_seed: int,
    sources: Sequence[str],
    context_before: int,
    context_after: int,
    min_sentence_chars: int,
    max_sentence_chars: int,
    min_edit_similarity: float,
    max_edit_similarity: float,
) -> pd.DataFrame:
    selected_keys = reservoir_sample_article_keys(
        connection,
        schema,
        max_articles=max_articles,
        seed=sampling_seed,
        sources=sources,
    )

    records: list[dict[str, Any]] = []
    processed_articles = 0

    for key, versions in load_selected_versions(connection, schema, selected_keys):
        processed_articles += 1
        article_records = extract_episodes_from_article(
            key,
            versions,
            context_before=context_before,
            context_after=context_after,
            min_sentence_chars=min_sentence_chars,
            max_sentence_chars=max_sentence_chars,
            min_edit_similarity=min_edit_similarity,
            max_edit_similarity=max_edit_similarity,
        )
        records.extend(article_records)

        if processed_articles % 1000 == 0:
            print(
                f"Processed articles={processed_articles:,}; "
                f"episodes={len(records):,}"
            )

        if max_episodes > 0 and len(records) >= max_episodes:
            records = records[:max_episodes]
            break

    if not records:
        raise ValueError(
            "No revision-lineage episodes were extracted. Run --inspect-only, "
            "check the article schema, or relax the edit/sentence filters."
        )

    frame = pd.DataFrame.from_records(records)
    frame = frame.drop_duplicates(subset=["episode_id"]).reset_index(drop=True)
    return frame


def save_episode_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".parquet"):
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise SystemExit(
                "Saving Parquet requires pyarrow. Install it or use .csv.gz."
            ) from exc
    else:
        compression = "gzip" if suffixes.endswith(".gz") else None
        frame.to_csv(path, index=False, compression=compression)


def load_episode_cache(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Mechanism-ablation feature sets and model
# ---------------------------------------------------------------------------


BASE_TEXT_COLUMNS = [
    "context_before",
    "retained_sentence",
    "context_after",
]
BASE_NUMERIC_COLUMNS = [
    "version_index",
    "n_versions",
    "sentence_position",
    "retained_chars",
    "retained_tokens",
]
BASE_CATEGORICAL_COLUMNS = ["source"]

REJECTED_TEXT_COLUMNS = ["rejected_sentence"]
EDIT_GEOMETRY_COLUMNS = [
    "rejected_chars",
    "rejected_tokens",
    "char_delta",
    "token_delta",
]
LEXICAL_RELATION_COLUMNS = [
    "edit_similarity",
    "lexical_jaccard",
]
ALL_TEXT_COLUMNS = BASE_TEXT_COLUMNS + REJECTED_TEXT_COLUMNS
ALL_PREFERENCE_NUMERIC_COLUMNS = (
    EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
)


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    name: str
    text_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    description: str
    uses_matched_shuffle: bool = False


def make_feature_specs(profile: str) -> list[FeatureSpec]:
    """Return coherent, baseline-conditioned mechanism ablations.

    Every mechanism model includes the complete current-text baseline. This
    makes each gain interpretable as incremental information beyond the
    retained sentence and article context.
    """
    base_text = tuple(BASE_TEXT_COLUMNS)
    base_num = tuple(BASE_NUMERIC_COLUMNS)
    base_cat = tuple(BASE_CATEGORICAL_COLUMNS)

    def spec(
        name: str,
        *,
        text: Sequence[str] = (),
        numeric: Sequence[str] = (),
        description: str,
        shuffled: bool = False,
    ) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            text_columns=base_text + tuple(text),
            numeric_columns=base_num + tuple(numeric),
            categorical_columns=base_cat,
            description=description,
            uses_matched_shuffle=shuffled,
        )

    specs = [
        spec(
            "baseline",
            description="Current context, retained sentence and metadata.",
        ),
        spec(
            "plus_edit_geometry",
            numeric=EDIT_GEOMETRY_COLUMNS,
            description=(
                "Baseline plus rejected length and edit-size geometry."
            ),
        ),
        spec(
            "plus_lexical_relation",
            numeric=LEXICAL_RELATION_COLUMNS,
            description=(
                "Baseline plus scalar similarity/overlap between the pair."
            ),
        ),
        spec(
            "plus_rejected_text",
            text=REJECTED_TEXT_COLUMNS,
            description="Baseline plus the authentic rejected words.",
        ),
        spec(
            "plus_text_geometry",
            text=REJECTED_TEXT_COLUMNS,
            numeric=EDIT_GEOMETRY_COLUMNS,
            description="Baseline plus rejected text and edit geometry.",
        ),
    ]

    if profile == "full":
        specs.extend(
            [
                spec(
                    "plus_text_lexical",
                    text=REJECTED_TEXT_COLUMNS,
                    numeric=LEXICAL_RELATION_COLUMNS,
                    description=(
                        "Baseline plus rejected text and lexical relation."
                    ),
                ),
                spec(
                    "plus_geometry_lexical",
                    numeric=(
                        EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
                    ),
                    description=(
                        "Baseline plus all non-text preference evidence."
                    ),
                ),
            ]
        )

    specs.extend(
        [
            spec(
                "full_preference",
                text=REJECTED_TEXT_COLUMNS,
                numeric=(
                    EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
                ),
                description=(
                    "Baseline plus authentic rejected text, geometry and "
                    "lexical relation."
                ),
            ),
            spec(
                "matched_shuffled_full_preference",
                text=REJECTED_TEXT_COLUMNS,
                numeric=(
                    EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
                ),
                description=(
                    "Full bundle after a local matched shuffle of rejected "
                    "text, with every pair-derived metric recomputed."
                ),
                shuffled=True,
            ),
        ]
    )
    return specs


def build_model_pipeline(
    *,
    feature_spec: FeatureSpec,
    seed: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    logistic_c: float,
    logistic_solver: str,
    logistic_max_iter: int,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, Any]] = []

    for column in feature_spec.text_columns:
        transformers.append(
            (
                f"text_{column}",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=tfidf_min_df,
                    max_df=1.0,
                    max_features=tfidf_max_features,
                    sublinear_tf=True,
                    norm="l2",
                ),
                column,
            )
        )

    if feature_spec.numeric_columns:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipe, list(feature_spec.numeric_columns))
        )

    if feature_spec.categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(feature_spec.categorical_columns),
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.1,
    )

    classifier = LogisticRegression(
        C=logistic_c,
        l1_ratio=0.0,
        solver=logistic_solver,
        max_iter=logistic_max_iter,
        tol=1e-4,
        class_weight=None,
        random_state=seed,
    )

    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_spec: FeatureSpec,
    target_column: str,
    seed: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    logistic_c: float,
    logistic_solver: str,
    logistic_max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    train_model_df = train_df.copy()
    test_model_df = test_df.copy()
    for column in set(feature_spec.text_columns):
        train_model_df[column] = (
            train_model_df[column]
            .fillna("")
            .astype(str)
            .map(normalise_space)
            .replace("", "__EMPTY__")
        )
        test_model_df[column] = (
            test_model_df[column]
            .fillna("")
            .astype(str)
            .map(normalise_space)
            .replace("", "__EMPTY__")
        )

    y_train = train_model_df[target_column].astype(int).to_numpy()
    y_test = test_model_df[target_column].astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        raise ValueError("The training split contains only one target class.")

    pipeline = build_model_pipeline(
        feature_spec=feature_spec,
        seed=seed,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        logistic_c=logistic_c,
        logistic_solver=logistic_solver,
        logistic_max_iter=logistic_max_iter,
    )
    pipeline.fit(train_model_df, y_train)

    probabilities = pipeline.predict_proba(test_model_df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    train_prevalence = float(y_train.mean())
    test_prevalence = float(y_test.mean())
    null_probabilities = np.full(
        len(y_test), np.clip(train_prevalence, 1e-12, 1.0 - 1e-12)
    )
    iterations = int(np.max(pipeline.named_steps["classifier"].n_iter_))

    metrics = {
        "loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "train_prevalence": train_prevalence,
        "test_prevalence": test_prevalence,
        "mean_predicted_probability": float(probabilities.mean()),
        "probability_min": float(np.min(probabilities)),
        "probability_p01": float(np.quantile(probabilities, 0.01)),
        "probability_p05": float(np.quantile(probabilities, 0.05)),
        "probability_median": float(np.median(probabilities)),
        "probability_p95": float(np.quantile(probabilities, 0.95)),
        "probability_p99": float(np.quantile(probabilities, 0.99)),
        "probability_max": float(np.max(probabilities)),
        "calibration_gap": float(probabilities.mean() - test_prevalence),
        "solver": logistic_solver,
        "converged": bool(iterations < logistic_max_iter),
        "n_iter": iterations,
        "null_log_loss": float(
            log_loss(y_test, null_probabilities, labels=[0, 1])
        ),
        "null_brier": float(brier_score_loss(y_test, null_probabilities)),
    }
    return probabilities, metrics


def _local_matched_donor_indices(
    frame: pd.DataFrame,
    *,
    seed: int,
    block_size: int,
) -> np.ndarray:
    """Create a no-self local permutation in structural-neighbour blocks.

    Rows are sorted by source and edit geometry before being partitioned into
    small blocks. Each block receives a non-zero cyclic shift. This is not an
    exact conditional randomisation test, but it produces a substantially
    harder and more coherent semantic control than an unrestricted shuffle.
    """
    if block_size < 2:
        raise ValueError("matched shuffle block size must be at least 2")

    rng = np.random.default_rng(seed)
    donor = np.arange(len(frame), dtype=int)
    working = frame.reset_index(drop=False).rename(columns={"index": "_row"})

    sort_columns = [
        "source",
        "retained_chars",
        "rejected_chars",
        "edit_similarity",
        "lexical_jaccard",
        "version_index",
        "sentence_position",
    ]
    working = working.sort_values(sort_columns, kind="mergesort")

    for _source, source_part in working.groupby("source", sort=False):
        positions = source_part["_row"].to_numpy(dtype=int)
        if len(positions) <= 1:
            continue

        blocks = [
            positions[start : start + block_size]
            for start in range(0, len(positions), block_size)
        ]
        if len(blocks) > 1 and len(blocks[-1]) == 1:
            blocks[-2] = np.concatenate([blocks[-2], blocks[-1]])
            blocks.pop()

        for block in blocks:
            if len(block) <= 1:
                continue
            shift = int(rng.integers(1, len(block)))
            donor[block] = np.roll(block, shift)

    return donor


def matched_shuffle_rejected_text(
    frame: pd.DataFrame,
    *,
    seed: int,
    block_size: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Shuffle rejected text locally and recompute every derived pair metric."""
    shuffled = frame.reset_index(drop=True).copy()
    donor = _local_matched_donor_indices(
        shuffled, seed=seed, block_size=block_size
    )

    original_rejected = shuffled["rejected_sentence"].fillna("").astype(str)
    donor_rejected = original_rejected.iloc[donor].reset_index(drop=True)
    shuffled["rejected_sentence"] = donor_rejected

    shuffled["rejected_chars"] = donor_rejected.str.len().astype(int)
    shuffled["rejected_tokens"] = donor_rejected.map(
        lambda value: len(tokenise(value))
    ).astype(int)
    shuffled["char_delta"] = (
        shuffled["retained_chars"].astype(int)
        - shuffled["rejected_chars"].astype(int)
    )
    shuffled["token_delta"] = (
        shuffled["retained_tokens"].astype(int)
        - shuffled["rejected_tokens"].astype(int)
    )
    shuffled["edit_similarity"] = [
        difflib.SequenceMatcher(
            a=normalise_sentence(rejected),
            b=normalise_sentence(retained),
            autojunk=False,
        ).ratio()
        for rejected, retained in zip(
            shuffled["rejected_sentence"], shuffled["retained_sentence"]
        )
    ]
    shuffled["lexical_jaccard"] = [
        lexical_jaccard(rejected, retained)
        for rejected, retained in zip(
            shuffled["rejected_sentence"], shuffled["retained_sentence"]
        )
    ]

    same_text = np.asarray(
        [
            normalise_sentence(a) == normalise_sentence(b)
            for a, b in zip(original_rejected, donor_rejected)
        ],
        dtype=float,
    )
    diagnostics = {
        "same_rejected_text_rate": float(same_text.mean()),
        "mean_abs_rejected_chars_change": float(
            np.mean(
                np.abs(
                    frame.reset_index(drop=True)["rejected_chars"].to_numpy()
                    - shuffled["rejected_chars"].to_numpy()
                )
            )
        ),
        "mean_abs_similarity_change": float(
            np.mean(
                np.abs(
                    frame.reset_index(drop=True)["edit_similarity"].to_numpy()
                    - shuffled["edit_similarity"].to_numpy()
                )
            )
        ),
    }
    return shuffled, diagnostics


def build_group_loss_stats(
    test_df: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    y_true = test_df[target_column].astype(int).to_numpy()
    row_stats = pd.DataFrame(
        {
            "group_id": test_df["article_key"].astype(str).to_numpy(),
            "n_rows": 1,
        },
        index=test_df.index,
    )
    for name, probability in predictions.items():
        row_stats[f"{name}__log_sum"] = log_loss_components(
            y_true, probability
        )
        row_stats[f"{name}__brier_sum"] = brier_components(
            y_true, probability
        )
    grouped = row_stats.groupby("group_id", as_index=False).sum(
        numeric_only=True
    )
    grouped["seed"] = seed
    return grouped


def evaluate_seed(
    frame: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float,
    target_column: str,
    tfidf_max_features: int,
    tfidf_min_df: int,
    logistic_c: float,
    logistic_solver: str,
    logistic_max_iter: int,
    ablation_profile: str,
    matched_shuffle_block_size: int,
) -> EvaluationBundle:
    rng = np.random.default_rng(seed)
    groups = np.asarray(
        sorted(frame["article_key"].dropna().astype(str).unique())
    )
    if len(groups) < 2:
        raise ValueError("At least two article groups are required.")
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    n_test = min(n_test, len(groups) - 1)
    test_groups = set(groups[:n_test])

    group_values = frame["article_key"].astype(str)
    train_df = frame[~group_values.isin(test_groups)].copy().reset_index(drop=True)
    test_df = frame[group_values.isin(test_groups)].copy().reset_index(drop=True)

    shuffled_train, train_shuffle_diagnostics = matched_shuffle_rejected_text(
        train_df,
        seed=seed + 10_000,
        block_size=matched_shuffle_block_size,
    )
    shuffled_test, test_shuffle_diagnostics = matched_shuffle_rejected_text(
        test_df,
        seed=seed + 20_000,
        block_size=matched_shuffle_block_size,
    )
    print(
        "Matched-shuffle diagnostics: "
        f"train_same={train_shuffle_diagnostics['same_rejected_text_rate']:.4f}; "
        f"test_same={test_shuffle_diagnostics['same_rejected_text_rate']:.4f}; "
        f"test_mean_abs_similarity_change="
        f"{test_shuffle_diagnostics['mean_abs_similarity_change']:.4f}"
    )

    rows: list[ResultRow] = []
    predictions: dict[str, np.ndarray] = {}
    for feature_spec in make_feature_specs(ablation_profile):
        model_train = shuffled_train if feature_spec.uses_matched_shuffle else train_df
        model_test = shuffled_test if feature_spec.uses_matched_shuffle else test_df
        probability, metrics = fit_and_score(
            model_train,
            model_test,
            feature_spec=feature_spec,
            target_column=target_column,
            seed=seed,
            tfidf_max_features=tfidf_max_features,
            tfidf_min_df=tfidf_min_df,
            logistic_c=logistic_c,
            logistic_solver=logistic_solver,
            logistic_max_iter=logistic_max_iter,
        )
        predictions[feature_spec.name] = probability
        rows.append(
            ResultRow(
                track="newsedits",
                condition="sentence_revision_mechanism_ablation",
                seed=seed,
                target=target_column,
                feature_set=feature_spec.name,
                n_train=len(model_train),
                n_test=len(model_test),
                n_train_groups=model_train["article_key"].nunique(),
                n_test_groups=model_test["article_key"].nunique(),
                **metrics,
            )
        )

    stats = build_group_loss_stats(
        test_df,
        target_column=target_column,
        seed=seed,
        predictions=predictions,
    )
    return EvaluationBundle(rows=rows, group_loss_stats=stats)


# ---------------------------------------------------------------------------
# Generic paired comparisons and hierarchical bootstrap
# ---------------------------------------------------------------------------


def comparison_definitions(available: set[str]) -> list[tuple[str, str, str]]:
    requested = [
        ("full_vs_baseline", "baseline", "full_preference"),
        ("geometry_vs_baseline", "baseline", "plus_edit_geometry"),
        ("lexical_vs_baseline", "baseline", "plus_lexical_relation"),
        ("rejected_text_vs_baseline", "baseline", "plus_rejected_text"),
        ("text_geometry_vs_baseline", "baseline", "plus_text_geometry"),
        ("text_lexical_vs_baseline", "baseline", "plus_text_lexical"),
        (
            "geometry_lexical_vs_baseline",
            "baseline",
            "plus_geometry_lexical",
        ),
        (
            "semantic_increment_beyond_nontext",
            "plus_geometry_lexical",
            "full_preference",
        ),
        (
            "geometry_increment_beyond_text_lexical",
            "plus_text_lexical",
            "full_preference",
        ),
        (
            "lexical_increment_beyond_text_geometry",
            "plus_text_geometry",
            "full_preference",
        ),
        (
            "authentic_vs_matched_shuffle",
            "matched_shuffled_full_preference",
            "full_preference",
        ),
    ]
    return [item for item in requested if item[1] in available and item[2] in available]


def hierarchical_bootstrap_comparison(
    stats: pd.DataFrame,
    *,
    reference_feature_set: str,
    candidate_feature_set: str,
    metric: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")
    if metric not in {"log_loss", "brier"}:
        raise ValueError(metric)

    suffix = "log_sum" if metric == "log_loss" else "brier_sum"
    reference_column = f"{reference_feature_set}__{suffix}"
    candidate_column = f"{candidate_feature_set}__{suffix}"
    columns = ["n_rows", reference_column, candidate_column]

    rng = np.random.default_rng(seed)
    seed_values = np.asarray(sorted(stats["seed"].unique()), dtype=int)
    by_seed = {
        int(seed_value): stats.loc[
            stats["seed"] == seed_value, columns
        ].to_numpy(dtype=float)
        for seed_value in seed_values
    }

    draws = np.empty(samples, dtype=float)
    for draw_index in range(samples):
        sampled_seeds = rng.choice(
            seed_values, size=len(seed_values), replace=True
        )
        totals = np.zeros(3, dtype=float)
        for sampled_seed in sampled_seeds:
            array = by_seed[int(sampled_seed)]
            indices = rng.integers(0, len(array), size=len(array))
            totals += array[indices].sum(axis=0)
        draws[draw_index] = (totals[1] - totals[2]) / totals[0]

    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def build_summary_rows(
    rows: list[ResultRow],
    group_stats: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> list[SummaryRow]:
    frame = pd.DataFrame([dataclasses.asdict(row) for row in rows])
    available = set(frame["feature_set"].unique())
    summaries: list[SummaryRow] = []

    for comparison, reference, candidate in comparison_definitions(available):
        for metric, result_column in [("log_loss", "loss"), ("brier", "brier")]:
            per_seed: list[float] = []
            for seed_value in sorted(frame["seed"].unique()):
                indexed = frame[frame["seed"] == seed_value].set_index(
                    "feature_set"
                )
                per_seed.append(
                    float(
                        indexed.loc[reference, result_column]
                        - indexed.loc[candidate, result_column]
                    )
                )
            values = np.asarray(per_seed, dtype=float)
            low, high = hierarchical_bootstrap_comparison(
                group_stats,
                reference_feature_set=reference,
                candidate_feature_set=candidate,
                metric=metric,
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=(
                    bootstrap_seed
                    + sum(ord(char) for char in comparison + metric)
                ),
            )
            summaries.append(
                SummaryRow(
                    track="newsedits",
                    condition="sentence_revision_mechanism_ablation",
                    comparison=comparison,
                    metric=metric,
                    reference_feature_set=reference,
                    candidate_feature_set=candidate,
                    n_seeds=len(values),
                    mean_gain=float(values.mean()),
                    seed_std=(
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                    ci_low=low,
                    ci_high=high,
                    positive_seeds=int((values > 0).sum()),
                    confidence_level=confidence_level,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def audit_episodes(frame: pd.DataFrame) -> None:
    print_header("NewsEdits revision-lineage audit")
    print(f"Episodes: {len(frame):,}")
    print(f"Articles: {frame['article_key'].nunique():,}")
    print(f"Sources: {frame['source'].nunique():,}")
    print(
        "Target revised-again rate: "
        f"{frame['revised_again_next_version'].mean():.6f}"
    )
    print("\nTarget counts:")
    print(
        frame["revised_again_next_version"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )
    print("\nLargest sources:")
    print(frame["source"].value_counts().head(20).to_string())
    print("\nEdit similarity:")
    print(frame["edit_similarity"].describe().to_string())


def print_results(rows: list[ResultRow], summaries: list[SummaryRow]) -> None:
    result_frame = pd.DataFrame([dataclasses.asdict(row) for row in rows])
    aggregate = result_frame.groupby("feature_set")[[
        "loss",
        "brier",
        "auc",
        "average_precision",
        "accuracy",
        "mean_predicted_probability",
        "calibration_gap",
        "probability_p01",
        "probability_p99",
        "n_iter",
        "converged",
        "null_log_loss",
    ]].agg(["mean", "std"])

    print_header("Mechanism-ablation metrics across seeds")
    print(aggregate.to_string(float_format=lambda value: f"{value:.6f}"))

    sanity = result_frame.groupby("feature_set")[[
        "loss", "null_log_loss", "brier", "null_brier"
    ]].mean()
    sanity["log_loss_gain_vs_null"] = (
        sanity["null_log_loss"] - sanity["loss"]
    )
    sanity["brier_gain_vs_null"] = sanity["null_brier"] - sanity["brier"]
    print_header("Probability sanity check")
    print(sanity.to_string(float_format=lambda value: f"{value:.6f}"))

    summary_frame = pd.DataFrame(
        [dataclasses.asdict(row) for row in summaries]
    )
    print_header("Mechanism comparison summary")
    print(
        summary_frame[[
            "comparison",
            "metric",
            "reference_feature_set",
            "candidate_feature_set",
            "n_seeds",
            "mean_gain",
            "seed_std",
            "ci_low",
            "ci_high",
            "positive_seeds",
        ]].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate semantic, geometric and lexical mechanisms in NewsEdits "
            "preference-future forecasting."
        )
    )
    parser.add_argument("--db", help="Path to the NewsEdits SQLite database.")
    parser.add_argument(
        "--articles-table",
        default=None,
        help="Optional article-version table name; otherwise auto-discovered.",
    )
    parser.add_argument(
        "--split-table",
        default=None,
        help=(
            "Optional official split-sentence table name. "
            "Defaults to split_sentences when present."
        ),
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help=(
            "Source/outlet label for source-specific official databases. "
            "Defaults to a cleaned database filename."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print SQLite tables/schema and exit.",
    )
    parser.add_argument(
        "--episode-cache",
        default=None,
        help=(
            "CSV, CSV.GZ or Parquet path. Existing cache is loaded unless "
            "--rebuild-cache is supplied."
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=10_000,
        help="Reservoir-sampled articles with 3+ versions. Use 0 for all.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after this many episodes; 0 means no episode cap.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=1729,
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional source/outlet filter; may be repeated.",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min-sentence-chars",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max-sentence-chars",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--min-edit-similarity",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--max-edit-similarity",
        type=float,
        default=0.98,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated split/model seeds.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=40_000,
        help="Maximum TF-IDF features per text field.",
    )
    parser.add_argument(
        "--tfidf-min-df",
        type=int,
        default=2,
        help=(
            "Minimum document frequency per TF-IDF field. Use 2 or more "
            "to avoid one-off features destabilising small samples."
        ),
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=0.1,
        help=(
            "Inverse L2 regularisation strength. Smaller values are more "
            "regularised and usually better calibrated on small samples."
        ),
    )
    parser.add_argument(
        "--logistic-solver",
        choices=["auto", "liblinear", "saga"],
        default="auto",
        help=(
            "Use liblinear for small/medium binary samples and saga for "
            "large sparse samples. auto switches at 50,000 training rows."
        ),
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=10000,
        help="Maximum solver iterations; convergence is reported per model.",
    )
    parser.add_argument(
        "--ablation-profile",
        choices=["core", "full"],
        default="full",
        help=(
            "core runs the main mechanism models; full also runs the "
            "pairwise complementary ablations needed to estimate semantic, "
            "geometry and lexical increments."
        ),
    )
    parser.add_argument(
        "--matched-shuffle-block-size",
        type=int,
        default=32,
        help=(
            "Local structural-neighbour block size for the coherent rejected-"
            "text permutation control."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2718,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Per-seed result CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="PFI summary CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not 0.0 < args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be between 0 and 1.")
    if not 0.0 <= args.min_edit_similarity < args.max_edit_similarity <= 1.0:
        raise SystemExit("Edit similarity bounds must satisfy 0 <= min < max <= 1.")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative.")
    if args.tfidf_min_df < 1:
        raise SystemExit("--tfidf-min-df must be at least 1.")
    if args.logistic_c <= 0:
        raise SystemExit("--logistic-c must be positive.")
    if args.logistic_max_iter < 1:
        raise SystemExit("--logistic-max-iter must be at least 1.")
    if args.matched_shuffle_block_size < 2:
        raise SystemExit("--matched-shuffle-block-size must be at least 2.")

    cache_path = Path(args.episode_cache) if args.episode_cache else None

    if cache_path and cache_path.exists() and not args.rebuild_cache:
        print(f"Loading episode cache: {cache_path}")
        episodes = load_episode_cache(cache_path)
    else:
        if not args.db:
            raise SystemExit(
                "--db is required when an episode cache is not available."
            )

        db_path = Path(args.db)
        if not db_path.exists():
            raise SystemExit(f"SQLite database not found: {db_path}")

        connection = sqlite3.connect(str(db_path))
        try:
            if args.inspect_only:
                inspect_database(
                    connection,
                    args.articles_table,
                    args.split_table,
                )
                return 0

            try:
                split_schema = discover_split_sentence_schema(
                    connection,
                    args.split_table,
                )
            except ValueError:
                split_schema = None

            if split_schema is not None:
                source_name = infer_source_name(
                    db_path,
                    args.source_name,
                )
                print_header("Detected official split-sentence schema")
                print(dataclasses.asdict(split_schema))
                print(f"Source label: {source_name}")

                episodes = build_episode_dataframe_from_split_sentences(
                    connection,
                    split_schema,
                    source_name=source_name,
                    max_articles=args.max_articles,
                    max_episodes=args.max_episodes,
                    sampling_seed=args.sampling_seed,
                    context_before=args.context_before,
                    context_after=args.context_after,
                    min_sentence_chars=args.min_sentence_chars,
                    max_sentence_chars=args.max_sentence_chars,
                    min_edit_similarity=args.min_edit_similarity,
                    max_edit_similarity=args.max_edit_similarity,
                )
            else:
                schema = discover_article_schema(
                    connection,
                    args.articles_table,
                )
                print_header("Detected full-article schema")
                print(dataclasses.asdict(schema))

                episodes = build_episode_dataframe(
                    connection,
                    schema,
                    max_articles=args.max_articles,
                    max_episodes=args.max_episodes,
                    sampling_seed=args.sampling_seed,
                    sources=args.source,
                    context_before=args.context_before,
                    context_after=args.context_after,
                    min_sentence_chars=args.min_sentence_chars,
                    max_sentence_chars=args.max_sentence_chars,
                    min_edit_similarity=args.min_edit_similarity,
                    max_edit_similarity=args.max_edit_similarity,
                )
        finally:
            connection.close()

        if cache_path:
            save_episode_cache(episodes, cache_path)
            print(f"Saved episode cache: {cache_path}")

    required_columns = {
        "article_key",
        "source",
        "context_before",
        "retained_sentence",
        "context_after",
        "rejected_sentence",
        "revised_again_next_version",
        *BASE_NUMERIC_COLUMNS,
        *ALL_PREFERENCE_NUMERIC_COLUMNS,
    }
    missing = sorted(required_columns - set(episodes.columns))
    if missing:
        raise SystemExit(
            f"Episode cache is missing required columns: {missing}"
        )

    audit_episodes(episodes)

    seeds = parse_int_list(args.seeds, args.seed)
    if args.logistic_solver == "auto":
        estimated_train_rows = int(round(len(episodes) * (1.0 - args.test_fraction)))
        logistic_solver = (
            "liblinear" if estimated_train_rows < 50_000 else "saga"
        )
    else:
        logistic_solver = args.logistic_solver
    print(f"Logistic solver: {logistic_solver}")

    all_rows: list[ResultRow] = []
    all_stats: list[pd.DataFrame] = []

    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        bundle = evaluate_seed(
            episodes,
            seed=seed,
            test_fraction=args.test_fraction,
            target_column="revised_again_next_version",
            tfidf_max_features=args.tfidf_max_features,
            tfidf_min_df=args.tfidf_min_df,
            logistic_c=args.logistic_c,
            logistic_solver=logistic_solver,
            logistic_max_iter=args.logistic_max_iter,
            ablation_profile=args.ablation_profile,
            matched_shuffle_block_size=args.matched_shuffle_block_size,
        )
        all_rows.extend(bundle.rows)
        all_stats.append(bundle.group_loss_stats)

    unconverged = [
        row for row in all_rows if not row.converged
    ]
    if unconverged:
        names = sorted({row.feature_set for row in unconverged})
        print(
            "\nWARNING: Solver did not converge for: "
            + ", ".join(names)
            + ". Do not treat this run as final."
        )

    combined_stats = pd.concat(all_stats, ignore_index=True)
    summaries = build_summary_rows(
        all_rows,
        combined_stats,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )

    print_results(all_rows, summaries)

    if args.out:
        output = pd.DataFrame(
            [dataclasses.asdict(row) for row in all_rows]
        )
        output.to_csv(args.out, index=False)
        print(f"\nSaved per-seed results to {args.out}")

    if args.summary_out:
        summary_output = pd.DataFrame(
            [dataclasses.asdict(row) for row in summaries]
        )
        summary_output.to_csv(args.summary_out, index=False)
        print(f"Saved summary results to {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
