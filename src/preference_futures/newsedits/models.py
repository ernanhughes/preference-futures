"""Typed records for NewsEdits ingestion and lineage extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from preference_futures.episodes import PreferenceEpisode, RevisionTriplet, build_preference_episode

NEWSEDITS_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArticleSchema:
    """Resolved column names for a NewsEdits article-version table."""

    table: str
    source: str
    article_id: str
    version_id: str
    text: str
    created: str | None = None
    title: str | None = None
    num_versions: str | None = None


@dataclass(frozen=True, slots=True)
class ArticleVersion:
    """One full article version loaded from SQLite."""

    source: str
    article_id: str
    version_id: str
    text: str
    created: str | None = None
    title: str = ""

    @property
    def lineage_id(self) -> str:
        return f"{self.source}::{self.article_id}"


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Conservative controls for one-to-one sentence lineage extraction."""

    context_before: int = 1
    context_after: int = 1
    min_sentence_chars: int = 20
    max_sentence_chars: int = 500
    min_edit_similarity: float = 0.15
    max_edit_similarity: float = 0.98

    def __post_init__(self) -> None:
        if self.context_before < 0 or self.context_after < 0:
            raise ValueError("context sizes must be non-negative")
        if self.min_sentence_chars < 1:
            raise ValueError("min_sentence_chars must be positive")
        if self.max_sentence_chars < self.min_sentence_chars:
            raise ValueError("max_sentence_chars must be >= min_sentence_chars")
        if not 0.0 <= self.min_edit_similarity <= 1.0:
            raise ValueError("min_edit_similarity must be between 0 and 1")
        if not 0.0 <= self.max_edit_similarity <= 1.0:
            raise ValueError("max_edit_similarity must be between 0 and 1")
        if self.max_edit_similarity < self.min_edit_similarity:
            raise ValueError("max_edit_similarity must be >= min_edit_similarity")


class ExclusionReason(StrEnum):
    """Reasons a candidate lineage failed the extraction contract."""

    TOO_FEW_VERSIONS = "too_few_versions"
    EMPTY_VERSION = "empty_version"
    AMBIGUOUS_REPLACEMENT = "ambiguous_replacement"
    INVALID_REJECTED_SENTENCE = "invalid_rejected_sentence"
    INVALID_SELECTED_SENTENCE = "invalid_selected_sentence"
    FUTURE_UNRESOLVED = "future_unresolved"
    SIMILARITY_BELOW_MIN = "similarity_below_min"
    SIMILARITY_ABOVE_MAX = "similarity_above_max"


@dataclass(slots=True)
class ExtractionAudit:
    """Survival-funnel counts produced alongside extraction."""

    articles_seen: int = 0
    version_windows_seen: int = 0
    replacement_opcodes_seen: int = 0
    accepted_examples: int = 0
    exclusions: Counter[str] = field(default_factory=Counter)

    def exclude(self, reason: ExclusionReason, count: int = 1) -> None:
        self.exclusions[reason.value] += count

    def to_record(self) -> dict[str, object]:
        return {
            "articles_seen": self.articles_seen,
            "version_windows_seen": self.version_windows_seen,
            "replacement_opcodes_seen": self.replacement_opcodes_seen,
            "accepted_examples": self.accepted_examples,
            "exclusions": dict(sorted(self.exclusions.items())),
        }


@dataclass(frozen=True, slots=True)
class NewsEditsExample:
    """A canonical triplet plus NewsEdits provenance and decision context."""

    triplet: RevisionTriplet
    source: str
    article_id: str
    v0_version_id: str
    v1_version_id: str
    v2_version_id: str
    selected_sentence_index: int
    context_before: str
    context_after: str
    sentence_position: float
    edit_similarity: float
    lexical_jaccard: float

    def build_episode(self, *, seed: int) -> PreferenceEpisode:
        return build_preference_episode(self.triplet, seed=seed)

    def to_record(self, *, seed: int) -> dict[str, object]:
        episode = self.build_episode(seed=seed)
        return {
            **episode.to_record(),
            "newsedits_schema_version": NEWSEDITS_RECORD_SCHEMA_VERSION,
            "source": self.source,
            "article_id": self.article_id,
            "v0_version_id": self.v0_version_id,
            "v1_version_id": self.v1_version_id,
            "v2_version_id": self.v2_version_id,
            "selected_sentence_index": self.selected_sentence_index,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "sentence_position": self.sentence_position,
            "edit_similarity": self.edit_similarity,
            "lexical_jaccard": self.lexical_jaccard,
        }


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    examples: tuple[NewsEditsExample, ...]
    audit: ExtractionAudit
