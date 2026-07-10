"""Audit consistency between NewsEdits sentence alignment and future labels."""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.newsedits.database import connect_read_only, infer_source_name
from preference_futures.newsedits.extract import (
    extract_from_database,
    extract_from_split_database,
)
from preference_futures.newsedits.models import ExtractionConfig, NewsEditsExample
from preference_futures.newsedits.schema import (
    discover_article_schema,
    discover_split_sentence_schema,
)
from preference_futures.newsedits.text import normalise_sentence, normalise_space
from preference_futures.training.common import canonical_json_sha256, write_json

FUTURE_LABEL_AUDIT_SCHEMA_VERSION = 1


def run_future_label_integrity_audit(
    *,
    database_path: Path,
    output_directory: Path,
    table: str | None = None,
    split_table: str | None = None,
    source_name: str | None = None,
    seed: int = 0,
    max_articles: int = 0,
    max_examples: int = 0,
    sources: Sequence[str] = (),
    context_before: int = 1,
    context_after: int = 1,
    min_sentence_chars: int = 20,
    max_sentence_chars: int = 500,
    min_edit_similarity: float = 0.15,
    max_edit_similarity: float = 0.98,
    sample_limit: int = 50,
    force: bool = False,
) -> dict[str, Any]:
    """Re-extract examples and count labels inconsistent with alignment normalization."""

    database = database_path.expanduser().resolve()
    if not database.is_file():
        raise ValueError(f"NewsEdits database does not exist: {database}")
    output = output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"output directory is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if sample_limit < 0:
        raise ValueError("sample_limit cannot be negative")

    config = ExtractionConfig(
        context_before=context_before,
        context_after=context_after,
        min_sentence_chars=min_sentence_chars,
        max_sentence_chars=max_sentence_chars,
        min_edit_similarity=min_edit_similarity,
        max_edit_similarity=max_edit_similarity,
    )
    connection = connect_read_only(database)
    detected_format = "unknown"
    resolved_source_name: str | None = None
    try:
        article_schema = _try_discover(discover_article_schema, connection, table)
        if article_schema is not None:
            extraction = extract_from_database(
                connection,
                article_schema,
                config=config,
                max_articles=max_articles,
                max_examples=max_examples,
                seed=seed,
                sources=tuple(sources),
            )
            detected_format = "article_versions"
        else:
            preferred_split_table = split_table or table
            split_schema = _try_discover(
                discover_split_sentence_schema,
                connection,
                preferred_split_table,
            )
            if split_schema is None:
                raise ValueError("no supported NewsEdits schema was detected")
            resolved_source_name = infer_source_name(database, source_name)
            extraction = extract_from_split_database(
                connection,
                split_schema,
                source_name=resolved_source_name,
                config=config,
                max_articles=max_articles,
                max_examples=max_examples,
                seed=seed,
            )
            detected_format = "split_sentences"
    finally:
        connection.close()

    report = build_future_label_integrity_report(
        extraction.examples,
        database=database,
        input_format=detected_format,
        source_name=resolved_source_name,
        extraction_audit=extraction.audit.to_record(),
        sample_limit=sample_limit,
        config=config,
    )
    write_json(output / "future-label-integrity.json", report)
    (output / "future-label-integrity.md").write_text(
        render_future_label_integrity_markdown(report),
        encoding="utf-8",
    )
    return report


def build_future_label_integrity_report(
    examples: Sequence[NewsEditsExample],
    *,
    database: Path | None = None,
    input_format: str = "fixture",
    source_name: str | None = None,
    extraction_audit: Mapping[str, Any] | None = None,
    sample_limit: int = 50,
    config: ExtractionConfig | None = None,
) -> dict[str, Any]:
    """Build the pure report used by the database audit and unit tests."""

    categories: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    current_revised = 0
    corrected_revised = 0
    v2_missing = 0
    v2_present = 0
    for example in examples:
        v1 = example.triplet.v1_sentence
        v2 = example.triplet.v2_sentence
        current = bool(example.future_revised)
        current_revised += int(current)
        if v2 is None:
            v2_missing += 1
            corrected_revised += 1
            continue
        v2_present += 1
        corrected = normalise_sentence(v1) != normalise_sentence(v2)
        corrected_revised += int(corrected)
        if current and not corrected:
            category = categorise_normalisation_mismatch(v1, v2)
            categories[category] += 1
            if len(mismatches) < sample_limit:
                mismatches.append(
                    {
                        "episode_id": example.triplet.episode_id,
                        "lineage_id": example.triplet.lineage_id,
                        "category": category,
                        "v1_sentence": v1,
                        "v2_sentence": v2,
                    }
                )

    total = len(examples)
    mismatch_count = sum(categories.values())
    report: dict[str, Any] = {
        "future_label_audit_schema_version": FUTURE_LABEL_AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "database": None if database is None else str(database),
        "input_format": input_format,
        "source_name": source_name,
        "extraction_config": None
        if config is None
        else {
            "context_before": config.context_before,
            "context_after": config.context_after,
            "min_sentence_chars": config.min_sentence_chars,
            "max_sentence_chars": config.max_sentence_chars,
            "min_edit_similarity": config.min_edit_similarity,
            "max_edit_similarity": config.max_edit_similarity,
        },
        "extraction_audit": dict(extraction_audit or {}),
        "counts": {
            "episodes": total,
            "v2_present": v2_present,
            "v2_missing_or_unaligned": v2_missing,
            "current_future_revised": current_revised,
            "corrected_normalized_future_revised": corrected_revised,
            "normalization_mismatch_labels": mismatch_count,
        },
        "rates": {
            "current_future_revised": current_revised / max(1, total),
            "corrected_normalized_future_revised": corrected_revised / max(1, total),
            "mismatch_of_all_episodes": mismatch_count / max(1, total),
            "mismatch_of_current_revised": mismatch_count / max(1, current_revised),
            "mismatch_of_v2_present": mismatch_count / max(1, v2_present),
        },
        "mismatch_categories": dict(sorted(categories.items())),
        "sample_limit": sample_limit,
        "mismatch_samples": mismatches,
        "interpretation": {
            "primary_question": (
                "How often does the aligner treat V1 and V2 as the same sentence after "
                "case/quote normalization while the persisted future label calls it revised?"
            ),
            "changes_frozen_result": False,
            "requires_corrected_replication_if_material": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def categorise_normalisation_mismatch(left: str, right: str) -> str:
    """Explain which existing normalization operation makes two sentences equal."""

    left_space = normalise_space(left)
    right_space = normalise_space(right)
    if left_space == right_space:
        return "whitespace_only"
    if left_space.lower() == right_space.lower():
        return "case_only"
    left_quotes = _normalise_quotes(left_space)
    right_quotes = _normalise_quotes(right_space)
    if left_quotes == right_quotes:
        return "quote_style_only"
    if left_quotes.lower() == right_quotes.lower():
        return "case_and_quote_style"
    if normalise_sentence(left) == normalise_sentence(right):
        return "other_existing_normalization"
    return "not_explained_by_existing_normalization"


def render_future_label_integrity_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    rates = report["rates"]
    lines = [
        "# Step 7B Future-Label Integrity Audit",
        "",
        "**Status:** COMPLETE — EXPLORATORY",
        "",
        f"- Episodes: `{int(counts['episodes']):,}`",
        f"- Current revised labels: `{int(counts['current_future_revised']):,}`",
        (
            "- Revised under alignment normalization: "
            f"`{int(counts['corrected_normalized_future_revised']):,}`"
        ),
        (
            "- Inconsistent normalization labels: "
            f"`{int(counts['normalization_mismatch_labels']):,}`"
        ),
        (
            "- Mismatch rate of all episodes: "
            f"`{float(rates['mismatch_of_all_episodes']):.6%}`"
        ),
        (
            "- Mismatch rate of currently revised episodes: "
            f"`{float(rates['mismatch_of_current_revised']):.6%}`"
        ),
        "",
        "## Categories",
        "",
        "| Category | Episodes |",
        "|---|---:|",
    ]
    for category, count in report["mismatch_categories"].items():
        lines.append(f"| {category} | {int(count):,} |")
    if not report["mismatch_categories"]:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "This audit does not alter the frozen Steps 1-6 result. It quantifies whether a",
            "corrected replication should rebuild the future target with the same normalization",
            "rule used by sentence alignment.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalise_quotes(value: str) -> str:
    return value.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def _try_discover(discoverer: Any, connection: Any, preferred_table: str | None) -> Any | None:
    try:
        return discoverer(connection, preferred_table)
    except ValueError:
        return None
