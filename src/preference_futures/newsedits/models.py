"""Typed records for NewsEdits ingestion and lineage extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from preference_futures.episodes import PreferenceEpisode, RevisionTriplet, build_preference_episode

NEWSEDITS_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArticleSchema:
    """Resolved column names for a full article-version table."""

    table: str
    source: str
    article_id: str
    version_id: str
    text: str
    created: str | None = None
    title: str | None = None
    num_versions: str | None = None


@dataclass(frozen=True, slots=True)
class SplitSentenceSchema:
    """Resolved columns for the official NewsEdits ``split_sentences`` table."""

    table: str
    article_id: str
    version_id: str
    sentence_id: str
    sentence: str


@dataclass(frozen=True, slots=True)
class ArticleVersion:
    """One article version loaded or reconstructed from SQLite."""

    source: str
    article_id: str
    version_id: str
    text: str
    sentences: tuple[str, ...] | None = None
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
    """Survival-funnel and target-balance counts produced alongside extraction."""

    articles_seen: int = 0
    articles_with_examples: int = 0
    version_windows_seen: int = 0
    replacement_opcodes_seen: int = 0
    accepted_examples: int = 0
    future_revised_examples: int = 0
    future_stable_examples: int = 0
    exclusions: Counter[str] = field(default_factory=Counter)

    def exclude(self, reason: ExclusionReason, count: int = 1) -> None:
        self.exclusions[reason.value] += count

    def finalize(self, examples: Sequence[NewsEditsExample]) -> None:
        """Derive final counts after duplicate removal and output limiting."""

        self.accepted_examples = len(examples)
        self.articles_with_examples = len({example.triplet.lineage_id for example in examples})
        self.future_revised_examples = sum(example.future_revised for example in examples)
        self.future_stable_examples = self.accepted_examples - self.future_revised_examples

    def to_record(self) -> dict[str, object]:
        acceptance_rate = (
            self.accepted_examples / self.replacement_opcodes_seen
            if self.replacement_opcodes_seen
            else 0.0
        )
        future_revised_rate = (
            self.future_revised_examples / self.accepted_examples
            if self.accepted_examples
            else 0.0
        )
        return {
            "articles_seen": self.articles_seen,
            "articles_with_examples": self.articles_with_examples,
            "version_windows_seen": self.version_windows_seen,
            "replacement_opcodes_seen": self.replacement_opcodes_seen,
            "accepted_examples": self.accepted_examples,
            "acceptance_rate": acceptance_rate,
            "future_revised_examples": self.future_revised_examples,
            "future_stable_examples": self.future_stable_examples,
            "future_revised_rate": future_revised_rate,
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

    @property
    def future_revised(self) -> bool:
        """Return whether the selected V1 sentence changed or disappeared in V2."""

        future = self.triplet.v2_sentence
        return future is None or " ".join(future.split()) != " ".join(
            self.triplet.v1_sentence.split()
        )

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
