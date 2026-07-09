"""Extract generic temporal-direction pairs from disjoint NewsEdits lineages."""

from __future__ import annotations

import difflib
import hashlib
import random
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

_ARTICLE_ALIASES = ("entry_id", "a_id", "article_id", "articleid", "story_id")
_VERSION_ALIASES = ("version", "version_id", "v_id", "revision_id")
_SENTENCE_ID_ALIASES = ("sent_idx", "sentence_id", "sent_id", "sentence_index")
_SENTENCE_ALIASES = ("sentence", "sent", "text")


def extract_independent_temporal_pairs(
    database_path: Path,
    *,
    excluded_lineages: set[str],
    source_name: str,
    target_pairs: int,
    seed: int = 17,
    max_articles: int = 20000,
    min_sentence_chars: int = 20,
    max_sentence_chars: int = 500,
    min_similarity: float = 0.05,
    max_similarity: float = 0.995,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract one-to-one replacements from articles excluded from future evaluation."""

    if target_pairs < 1:
        raise ValueError("target_pairs must be positive")
    connection = _connect_read_only(database_path)
    try:
        schema = discover_split_sentence_schema(connection)
        excluded_ids = {_article_id(lineage) for lineage in excluded_lineages}
        article_ids = _sample_external_article_ids(
            connection,
            schema,
            excluded_ids=excluded_ids,
            max_articles=max_articles,
            seed=seed,
        )
        pairs: list[dict[str, Any]] = []
        seen_text_pairs: set[tuple[str, str]] = set()
        articles_seen = 0
        replacement_opcodes = 0
        for article_id, versions in _iter_versions(connection, schema, article_ids):
            articles_seen += 1
            for old_version, new_version in zip(versions, versions[1:]):
                old_sentences = old_version[1]
                new_sentences = new_version[1]
                matcher = difflib.SequenceMatcher(
                    a=[_normalise(value) for value in old_sentences],
                    b=[_normalise(value) for value in new_sentences],
                    autojunk=False,
                )
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != "replace":
                        continue
                    replacement_opcodes += 1
                    if (i2 - i1) != 1 or (j2 - j1) != 1:
                        continue
                    earlier = _space(old_sentences[i1])
                    later = _space(new_sentences[j1])
                    if not _valid(earlier, min_sentence_chars, max_sentence_chars):
                        continue
                    if not _valid(later, min_sentence_chars, max_sentence_chars):
                        continue
                    similarity = difflib.SequenceMatcher(
                        a=_normalise(earlier),
                        b=_normalise(later),
                        autojunk=False,
                    ).ratio()
                    if not min_similarity <= similarity <= max_similarity:
                        continue
                    text_key = (_normalise(earlier), _normalise(later))
                    if text_key in seen_text_pairs:
                        continue
                    seen_text_pairs.add(text_key)
                    before = " ".join(new_sentences[max(0, j1 - 1) : j1])
                    after = " ".join(new_sentences[j1 + 1 : j1 + 2])
                    lineage_id = f"{source_name}::{article_id}"
                    pair_id = (
                        f"{lineage_id}::{old_version[0]}->{new_version[0]}::{j1}::temporal"
                    )
                    pairs.append(
                        {
                            "temporal_pair_schema_version": 1,
                            "temporal_pair_id": pair_id,
                            "lineage_id": lineage_id,
                            "earlier_version_id": old_version[0],
                            "later_version_id": new_version[0],
                            "earlier_text": earlier,
                            "later_text": later,
                            "context_before": before,
                            "context_after": after,
                            "edit_similarity": similarity,
                        }
                    )
                    if len(pairs) >= target_pairs:
                        return pairs, _audit(
                            database_path,
                            schema,
                            target_pairs,
                            pairs,
                            article_ids,
                            articles_seen,
                            replacement_opcodes,
                            excluded_lineages,
                            seed,
                        )
        raise ValueError(
            f"only {len(pairs)} independent temporal pairs were extracted; "
            f"{target_pairs} are required. Increase --temporal-max-articles."
        )
    finally:
        connection.close()


def write_temporal_pairs(
    path: Path,
    audit_path: Path,
    pairs: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json_dumps(pair) + "\n")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json_dumps(audit, indent=2) + "\n", encoding="utf-8")


def discover_split_sentence_schema(connection: sqlite3.Connection) -> dict[str, str]:
    tables = [
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    candidates: list[dict[str, str]] = []
    for table in tables:
        columns = [
            row[1] for row in connection.execute(f"PRAGMA table_info({_quote(table)})")
        ]
        lowered = {column.casefold(): column for column in columns}
        mapping = {
            "table": table,
            "article_id": _alias(lowered, _ARTICLE_ALIASES),
            "version_id": _alias(lowered, _VERSION_ALIASES),
            "sentence_id": _alias(lowered, _SENTENCE_ID_ALIASES),
            "sentence": _alias(lowered, _SENTENCE_ALIASES),
        }
        if all(
            mapping[field]
            for field in ("article_id", "version_id", "sentence_id", "sentence")
        ):
            candidates.append({key: str(value) for key, value in mapping.items()})
    if not candidates:
        raise ValueError("could not discover a split_sentences-compatible table")
    candidates.sort(key=lambda value: (value["table"] != "split_sentences", value["table"]))
    return candidates[0]


def _sample_external_article_ids(
    connection: sqlite3.Connection,
    schema: Mapping[str, str],
    *,
    excluded_ids: set[str],
    max_articles: int,
    seed: int,
) -> list[tuple[Any, int]]:
    table = _quote(schema["table"])
    article = _quote(schema["article_id"])
    version = _quote(schema["version_id"])
    sentence = _quote(schema["sentence"])
    sql = f"""
        SELECT {article}, COUNT(DISTINCT {version})
        FROM {table}
        WHERE {sentence} IS NOT NULL AND LENGTH(TRIM({sentence})) > 0
        GROUP BY {article}
        HAVING COUNT(DISTINCT {version}) >= 2
    """
    rng = random.Random(seed)
    reservoir: list[tuple[Any, int]] = []
    seen = 0
    for article_id, version_count in connection.execute(sql):
        if str(article_id) in excluded_ids:
            continue
        seen += 1
        item = (article_id, int(version_count))
        if max_articles <= 0 or len(reservoir) < max_articles:
            reservoir.append(item)
            continue
        replacement = rng.randrange(seen)
        if replacement < max_articles:
            reservoir[replacement] = item
    return reservoir


def _iter_versions(
    connection: sqlite3.Connection,
    schema: Mapping[str, str],
    selected_ids: Sequence[tuple[Any, int]],
) -> Iterator[tuple[str, list[tuple[str, list[str]]]]]:
    connection.execute("DROP TABLE IF EXISTS temp.pf_temporal_entries")
    connection.execute(
        "CREATE TEMP TABLE pf_temporal_entries (article_value, version_count INTEGER)"
    )
    connection.executemany("INSERT INTO pf_temporal_entries VALUES (?, ?)", selected_ids)
    table = _quote(schema["table"])
    article = _quote(schema["article_id"])
    version = _quote(schema["version_id"])
    sentence_id = _quote(schema["sentence_id"])
    sentence = _quote(schema["sentence"])
    sql = f"""
        SELECT s.{article}, s.{version}, s.{sentence_id}, s.{sentence}
        FROM {table} AS s
        INNER JOIN temp.pf_temporal_entries AS k ON s.{article} = k.article_value
        WHERE s.{sentence} IS NOT NULL AND LENGTH(TRIM(s.{sentence})) > 0
        ORDER BY s.{article}, CAST(s.{version} AS REAL), s.{version},
                 CAST(s.{sentence_id} AS INTEGER), s.{sentence_id}
    """
    current_article: str | None = None
    current_version: str | None = None
    sentences: list[str] = []
    versions: list[tuple[str, list[str]]] = []

    def flush_version() -> None:
        nonlocal sentences
        if current_version is not None and sentences:
            versions.append((current_version, list(sentences)))
        sentences = []

    for article_value, version_value, _sentence_index, sentence_value in connection.execute(sql):
        article_value = str(article_value)
        version_value = str(version_value)
        if current_article is None:
            current_article = article_value
            current_version = version_value
        elif article_value != current_article:
            flush_version()
            yield current_article, versions
            current_article = article_value
            current_version = version_value
            versions = []
        elif version_value != current_version:
            flush_version()
            current_version = version_value
        value = _space(str(sentence_value))
        if value:
            sentences.append(value)
    if current_article is not None:
        flush_version()
        if versions:
            yield current_article, versions


def _audit(
    database_path: Path,
    schema: Mapping[str, str],
    target_pairs: int,
    pairs: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[tuple[Any, int]],
    articles_seen: int,
    replacement_opcodes: int,
    excluded_lineages: set[str],
    seed: int,
) -> dict[str, Any]:
    temporal_lineages = {str(pair["lineage_id"]) for pair in pairs}
    overlap = temporal_lineages & excluded_lineages
    return {
        "temporal_pair_audit_schema_version": 1,
        "database": _source_metadata(database_path),
        "schema": dict(schema),
        "seed": seed,
        "target_pairs": target_pairs,
        "accepted_pairs": len(pairs),
        "selected_external_articles": len(selected_ids),
        "articles_seen_until_target": articles_seen,
        "replacement_opcodes_seen": replacement_opcodes,
        "temporal_lineages": len(temporal_lineages),
        "evaluation_lineage_overlap": len(overlap),
        "gates": {
            "target_pair_count_reached": len(pairs) >= target_pairs,
            "temporal_lineages_disjoint_from_evaluation": not overlap,
            "temporal_pair_ids_unique": (
                len({str(pair["temporal_pair_id"]) for pair in pairs}) == len(pairs)
            ),
        },
    }


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def _source_metadata(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _article_id(lineage_id: str) -> str:
    return lineage_id.split("::", 1)[1] if "::" in lineage_id else lineage_id


def _alias(columns: Mapping[str, str], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        if alias.casefold() in columns:
            return columns[alias.casefold()]
    return None


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _space(value: str) -> str:
    return " ".join(value.split())


def _normalise(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def _valid(value: str, minimum: int, maximum: int) -> bool:
    return minimum <= len(value) <= maximum and any(character.isalpha() for character in value)


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)
