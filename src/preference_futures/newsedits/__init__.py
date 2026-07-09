"""Reusable NewsEdits ingestion and sentence-lineage extraction."""

from preference_futures.newsedits.database import (
    connect_read_only,
    infer_source_name,
    iter_article_versions,
    iter_split_article_versions,
    sample_article_keys,
    sample_split_article_ids,
)
from preference_futures.newsedits.extract import (
    extract_article_examples,
    extract_from_database,
    extract_from_split_database,
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
    SplitSentenceSchema,
)
from preference_futures.newsedits.schema import (
    discover_article_schema,
    discover_split_sentence_schema,
)

__all__ = [
    "NEWSEDITS_RECORD_SCHEMA_VERSION",
    "ArticleSchema",
    "ArticleVersion",
    "ExclusionReason",
    "ExtractionAudit",
    "ExtractionConfig",
    "ExtractionResult",
    "NewsEditsExample",
    "SplitSentenceSchema",
    "connect_read_only",
    "discover_article_schema",
    "discover_split_sentence_schema",
    "extract_article_examples",
    "extract_from_database",
    "extract_from_split_database",
    "infer_source_name",
    "iter_article_versions",
    "iter_split_article_versions",
    "sample_article_keys",
    "sample_split_article_ids",
    "sentence_future_map",
]
