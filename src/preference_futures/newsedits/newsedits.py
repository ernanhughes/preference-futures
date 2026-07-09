#!/usr/bin/env python3
r"""
PreferenceFutures — NewsEdits revision-lineage probe.

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

Positive held-out PFI means the rejected alternative carries information about
the next revision beyond the retained sentence itself.

This is a revealed-revision-preference experiment, not an explicit A/B-vote
experiment and not a causal estimate.

Expected NewsEdits source
-------------------------

The original NewsEdits release is described as SQLite tables. This script reads
the article-version table directly and discovers column names case-
insensitively. The expected logical fields are:

    SOURCE, A_ID, VERSION_ID, TEXT

Optional fields:

    CREATED, TITLE, NUM_VERSIONS

The script does not require the precomputed sentence_diffs table. It reconstructs
consecutive sentence revisions from full article versions, which makes the
future linkage explicit and avoids depending on tag-format details.

Dependencies
------------

    pip install pandas numpy scikit-learn

Optional, only for Parquet episode caches:

    pip install pyarrow

Examples
--------

Inspect the database schema:

    python preference_futures_newsedits.py \
      --db /path/to/newsedits.db \
      --inspect-only

Windows PowerShell smoke test:

    python preference_futures_newsedits.py `
      --db C:\data\newsedits.db `
      --max-articles 5000 `
      --max-episodes 50000 `
      --seeds 1,2,3 `
      --bootstrap-samples 500 `
      --episode-cache newsedits_smoke_episodes.csv.gz `
      --out newsedits_smoke_runs.csv `
      --summary-out newsedits_smoke_summary.csv

Larger run:

    python preference_futures_newsedits.py `
      --db C:\data\newsedits.db `
      --max-articles 100000 `
      --seeds 1,2,3,4,5,6,7,8,9,10 `
      --bootstrap-samples 5000 `
      --episode-cache newsedits_full_episodes.csv.gz `
      --out newsedits_full_runs.csv `
      --summary-out newsedits_full_summary.csv

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
    null_log_loss: float
    null_brier: float


@dataclasses.dataclass
class SummaryRow:
    track: str
    condition: str
    statistic: str
    n_seeds: int
    mean: float
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


def inspect_database(
    connection: sqlite3.Connection,
    preferred_table: str | None,
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

    print_header("Detected article schema")
    schema = discover_article_schema(connection, preferred_table)
    print(dataclasses.asdict(schema))


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
# Feature sets and model
# ---------------------------------------------------------------------------


BASE_TEXT_COLUMNS = [
    "context_before",
    "retained_sentence",
    "context_after",
]
PREFERENCE_TEXT_COLUMNS = ["rejected_sentence"]

BASE_NUMERIC_COLUMNS = [
    "version_index",
    "n_versions",
    "sentence_position",
    "retained_chars",
    "retained_tokens",
]
PREFERENCE_NUMERIC_COLUMNS = [
    "rejected_chars",
    "rejected_tokens",
    "char_delta",
    "token_delta",
    "edit_similarity",
    "lexical_jaccard",
]
BASE_CATEGORICAL_COLUMNS = ["source"]

PREFERENCE_BUNDLE_COLUMNS = (
    PREFERENCE_TEXT_COLUMNS + PREFERENCE_NUMERIC_COLUMNS
)


def build_model_pipeline(
    *,
    include_base: bool,
    include_preference: bool,
    seed: int,
    tfidf_max_features: int,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    transformers: list[tuple[str, Any, Any]] = []

    if include_base:
        for column in BASE_TEXT_COLUMNS:
            transformers.append(
                (
                    f"text_{column}",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=1.0,
                        max_features=tfidf_max_features,
                        sublinear_tf=True,
                    ),
                    column,
                )
            )

        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("base_num", numeric_pipe, BASE_NUMERIC_COLUMNS))
        transformers.append(
            (
                "base_cat",
                OneHotEncoder(handle_unknown="ignore"),
                BASE_CATEGORICAL_COLUMNS,
            )
        )

    if include_preference:
        transformers.append(
            (
                "text_rejected",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    max_features=tfidf_max_features,
                    sublinear_tf=True,
                ),
                "rejected_sentence",
            )
        )
        preference_numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(
            (
                "preference_num",
                preference_numeric_pipe,
                PREFERENCE_NUMERIC_COLUMNS,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.1,
    )

    # Scalable probabilistic linear baseline. No class reweighting: PFI uses
    # proper probability scoring rules and must preserve observed prevalence.
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-4,
        class_weight=None,
        random_state=seed,
        average=True,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    include_base: bool,
    include_preference: bool,
    target_column: str,
    seed: int,
    tfidf_max_features: int,
) -> tuple[np.ndarray, dict[str, float]]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    for text_column in BASE_TEXT_COLUMNS + PREFERENCE_TEXT_COLUMNS:
        if text_column in train_df.columns:
            train_df[text_column] = (
                train_df[text_column]
                .fillna("")
                .astype(str)
                .map(lambda value: value if value.strip() else "__empty__")
            )
            test_df[text_column] = (
                test_df[text_column]
                .fillna("")
                .astype(str)
                .map(lambda value: value if value.strip() else "__empty__")
            )

    y_train = train_df[target_column].astype(int).to_numpy()
    y_test = test_df[target_column].astype(int).to_numpy()

    if np.unique(y_train).size < 2:
        raise ValueError("The training split contains only one target class.")

    pipeline = build_model_pipeline(
        include_base=include_base,
        include_preference=include_preference,
        seed=seed,
        tfidf_max_features=tfidf_max_features,
    )
    pipeline.fit(train_df, y_train)

    probabilities = pipeline.predict_proba(test_df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    train_prevalence = float(y_train.mean())
    test_prevalence = float(y_test.mean())
    null_probabilities = np.full(
        len(y_test),
        np.clip(train_prevalence, 1e-12, 1.0 - 1e-12),
    )

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
        "null_log_loss": float(
            log_loss(y_test, null_probabilities, labels=[0, 1])
        ),
        "null_brier": float(
            brier_score_loss(y_test, null_probabilities)
        ),
    }
    return probabilities, metrics


def shuffle_preference_bundle(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Shuffle rejected-sentence evidence within source/length buckets."""
    shuffled = frame.copy()
    shuffled["_length_bucket"] = (
        shuffled["retained_chars"].fillna(0).astype(int) // 80
    ).clip(upper=20)

    pieces: list[pd.DataFrame] = []
    grouped = shuffled.groupby(
        ["source", "_length_bucket"],
        dropna=False,
        sort=False,
    )

    for group_index, (_key, part) in enumerate(grouped):
        if len(part) <= 1:
            values = part[PREFERENCE_BUNDLE_COLUMNS].to_numpy()
        else:
            values = part[PREFERENCE_BUNDLE_COLUMNS].sample(
                frac=1.0,
                random_state=seed + group_index,
            ).to_numpy()
        pieces.append(
            pd.DataFrame(
                values,
                columns=PREFERENCE_BUNDLE_COLUMNS,
                index=part.index,
            )
        )

    replacement = pd.concat(pieces).sort_index()
    for column in PREFERENCE_BUNDLE_COLUMNS:
        shuffled[column] = replacement[column].to_numpy()
    for column in PREFERENCE_NUMERIC_COLUMNS:
        shuffled[column] = pd.to_numeric(
            shuffled[column],
            errors="coerce",
        )
    return shuffled.drop(columns=["_length_bucket"])


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
        row_stats[f"{name}_log_sum"] = log_loss_components(
            y_true, probability
        )
        row_stats[f"{name}_brier_sum"] = brier_components(
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
    train_df = frame[~group_values.isin(test_groups)].copy()
    test_df = frame[group_values.isin(test_groups)].copy()

    configurations = {
        "context_retained_no_preference": (True, False),
        "rejected_preference_only": (False, True),
        "context_retained_plus_preference": (True, True),
    }

    rows: list[ResultRow] = []
    predictions: dict[str, np.ndarray] = {}

    for feature_set, (include_base, include_preference) in configurations.items():
        probability, metrics = fit_and_score(
            train_df,
            test_df,
            include_base=include_base,
            include_preference=include_preference,
            target_column=target_column,
            seed=seed,
            tfidf_max_features=tfidf_max_features,
        )
        predictions[feature_set] = probability
        rows.append(
            ResultRow(
                track="newsedits",
                condition="sentence_revision_stability",
                seed=seed,
                target=target_column,
                feature_set=feature_set,
                n_train=len(train_df),
                n_test=len(test_df),
                n_train_groups=train_df["article_key"].nunique(),
                n_test_groups=test_df["article_key"].nunique(),
                **metrics,
            )
        )

    shuffled_train = shuffle_preference_bundle(
        train_df,
        seed=seed + 10_000,
    )
    shuffled_test = shuffle_preference_bundle(
        test_df,
        seed=seed + 20_000,
    )
    shuffled_name = "context_retained_plus_shuffled_preference"
    shuffled_probability, shuffled_metrics = fit_and_score(
        shuffled_train,
        shuffled_test,
        include_base=True,
        include_preference=True,
        target_column=target_column,
        seed=seed,
        tfidf_max_features=tfidf_max_features,
    )
    predictions[shuffled_name] = shuffled_probability
    rows.append(
        ResultRow(
            track="newsedits",
            condition="sentence_revision_stability",
            seed=seed,
            target=target_column,
            feature_set=shuffled_name,
            n_train=len(shuffled_train),
            n_test=len(shuffled_test),
            n_train_groups=shuffled_train["article_key"].nunique(),
            n_test_groups=shuffled_test["article_key"].nunique(),
            **shuffled_metrics,
        )
    )

    stats = build_group_loss_stats(
        test_df,
        target_column=target_column,
        seed=seed,
        predictions={
            "no_pref": predictions["context_retained_no_preference"],
            "full": predictions["context_retained_plus_preference"],
            "shuffled": predictions[shuffled_name],
        },
    )
    return EvaluationBundle(rows=rows, group_loss_stats=stats)


# ---------------------------------------------------------------------------
# Aggregation and bootstrap
# ---------------------------------------------------------------------------


def hierarchical_bootstrap_interval(
    stats: pd.DataFrame,
    *,
    statistic: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    seed_values = np.asarray(sorted(stats["seed"].unique()), dtype=int)

    columns = [
        "n_rows",
        "no_pref_log_sum",
        "full_log_sum",
        "shuffled_log_sum",
        "no_pref_brier_sum",
        "full_brier_sum",
        "shuffled_brier_sum",
    ]
    by_seed = {
        int(seed_value): stats.loc[
            stats["seed"] == seed_value, columns
        ].to_numpy(dtype=float)
        for seed_value in seed_values
    }

    def compute(totals: np.ndarray) -> float:
        n_rows = totals[0]
        if statistic == "pfi_log_loss":
            return float((totals[1] - totals[2]) / n_rows)
        if statistic == "shuffle_gap_log_loss":
            return float((totals[3] - totals[2]) / n_rows)
        if statistic == "pfi_brier":
            return float((totals[4] - totals[5]) / n_rows)
        if statistic == "shuffle_gap_brier":
            return float((totals[6] - totals[5]) / n_rows)
        raise ValueError(statistic)

    draws = np.empty(samples, dtype=float)
    for draw_index in range(samples):
        sampled_seeds = rng.choice(
            seed_values,
            size=len(seed_values),
            replace=True,
        )
        totals = np.zeros(len(columns), dtype=float)
        for sampled_seed in sampled_seeds:
            array = by_seed[int(sampled_seed)]
            indices = rng.integers(0, len(array), size=len(array))
            totals += array[indices].sum(axis=0)
        draws[draw_index] = compute(totals)

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
    seed_values = sorted(frame["seed"].unique())

    values_by_statistic: dict[str, list[float]] = {
        "pfi_log_loss": [],
        "pfi_brier": [],
        "shuffle_gap_log_loss": [],
        "shuffle_gap_brier": [],
    }

    for seed in seed_values:
        indexed = frame[frame["seed"] == seed].set_index("feature_set")
        no_pref = indexed.loc["context_retained_no_preference"]
        full = indexed.loc["context_retained_plus_preference"]
        shuffled = indexed.loc[
            "context_retained_plus_shuffled_preference"
        ]

        values_by_statistic["pfi_log_loss"].append(
            float(no_pref["loss"] - full["loss"])
        )
        values_by_statistic["pfi_brier"].append(
            float(no_pref["brier"] - full["brier"])
        )
        values_by_statistic["shuffle_gap_log_loss"].append(
            float(shuffled["loss"] - full["loss"])
        )
        values_by_statistic["shuffle_gap_brier"].append(
            float(shuffled["brier"] - full["brier"])
        )

    summaries: list[SummaryRow] = []
    for statistic, values_list in values_by_statistic.items():
        values = np.asarray(values_list, dtype=float)
        low, high = hierarchical_bootstrap_interval(
            group_stats,
            statistic=statistic,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed + sum(ord(char) for char in statistic),
        )
        summaries.append(
            SummaryRow(
                track="newsedits",
                condition="sentence_revision_stability",
                statistic=statistic,
                n_seeds=len(values),
                mean=float(values.mean()),
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


def print_results(
    rows: list[ResultRow],
    summaries: list[SummaryRow],
) -> None:
    result_frame = pd.DataFrame(
        [dataclasses.asdict(row) for row in rows]
    )
    aggregate = result_frame.groupby("feature_set")[
        [
            "loss",
            "brier",
            "auc",
            "average_precision",
            "accuracy",
            "mean_predicted_probability",
        ]
    ].agg(["mean", "std"])

    print_header("Feature-set metrics across seeds")
    print(aggregate.to_string(float_format=lambda value: f"{value:.6f}"))

    summary_frame = pd.DataFrame(
        [dataclasses.asdict(row) for row in summaries]
    )
    print_header("PFI and shuffled-control summary")
    print(
        summary_frame[
            [
                "statistic",
                "n_seeds",
                "mean",
                "seed_std",
                "ci_low",
                "ci_high",
                "positive_seeds",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether rejected sentence alternatives improve forecasts "
            "of later sentence revision in NewsEdits."
        )
    )
    parser.add_argument("--db", help="Path to the NewsEdits SQLite database.")
    parser.add_argument(
        "--articles-table",
        default=None,
        help="Optional article-version table name; otherwise auto-discovered.",
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
                inspect_database(connection, args.articles_table)
                return 0

            schema = discover_article_schema(
                connection,
                args.articles_table,
            )
            print_header("Detected article schema")
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
        *PREFERENCE_NUMERIC_COLUMNS,
    }
    missing = sorted(required_columns - set(episodes.columns))
    if missing:
        raise SystemExit(
            f"Episode cache is missing required columns: {missing}"
        )

    audit_episodes(episodes)

    seeds = parse_int_list(args.seeds, args.seed)
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
        )
        all_rows.extend(bundle.rows)
        all_stats.append(bundle.group_loss_stats)

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
