"""Reusable NewsEdits ingestion and sentence-lineage extraction."""

from preference_futures.newsedits.database import (
    connect_read_only,
    iter_article_versions,
    sample_article_keys,
)
from preference_futures.newsedits.extract import (
    extract_article_examples,
    extract_from_database,
    sentence_future_map,
)
from preference_futures.newsedits.models import (
    NEWSEDITS_RECORD_SCHEMA_VERSION,
    ArticleSchema,
    ArticleVersion,
    ExclusionReason,
    ExtractionAudit,
    ExtractionConfig,
    ExtractionResult,
    NewsEditsExample,
)
from preference_futures.newsedits.schema import discover_article_schema

__all__ = [
    "NEWSEDITS_RECORD_SCHEMA_VERSION",
    "ArticleSchema",
    "ArticleVersion",
    "ExclusionReason",
    "ExtractionAudit",
    "ExtractionConfig",
    "ExtractionResult",
    "NewsEditsExample",
    "connect_read_only",
    "discover_article_schema",
    "extract_article_examples",
    "extract_from_database",
    "iter_article_versions",
    "sample_article_keys",
    "sentence_future_map",
]
