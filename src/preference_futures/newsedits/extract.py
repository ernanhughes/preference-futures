"""Sentence-lineage extraction from consecutive NewsEdits article versions."""

from __future__ import annotations

import difflib
import sqlite3
from collections.abc import Iterable, Sequence

from preference_futures.episodes import RevisionTriplet
from preference_futures.newsedits.database import (
    iter_article_versions,
    iter_split_article_versions,
    sample_article_keys,
    sample_split_article_ids,
)
from preference_futures.newsedits.models import (
    ArticleSchema,
    ArticleVersion,
    ExclusionReason,
    ExtractionAudit,
    ExtractionConfig,
    ExtractionResult,
    NewsEditsExample,
    SplitSentenceSchema,
)
from preference_futures.newsedits.text import (
    lexical_jaccard,
    normalise_sentence,
    sentence_split,
    surrounding_context,
    valid_sentence,
)


def sentence_future_map(
    middle_sentences: Sequence[str],
    future_sentences: Sequence[str],
) -> dict[int, str | None]:
    """Map each V1 sentence to an unambiguous V2 continuation, or ``None`` if changed."""

    matcher = difflib.SequenceMatcher(
        a=[normalise_sentence(value) for value in middle_sentences],
        b=[normalise_sentence(value) for value in future_sentences],
        autojunk=False,
    )
    mapped: dict[int, str | None] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, index in enumerate(range(i1, i2)):
                mapped[index] = future_sentences[j1 + offset]
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for offset, index in enumerate(range(i1, i2)):
                mapped[index] = future_sentences[j1 + offset]
        elif tag in {"replace", "delete"}:
            for index in range(i1, i2):
                mapped[index] = None
    return mapped


def extract_article_examples(
    versions: Sequence[ArticleVersion],
    *,
    config: ExtractionConfig | None = None,
    audit: ExtractionAudit | None = None,
) -> tuple[NewsEditsExample, ...]:
    """Extract conservative one-to-one V0→V1 choices and their V2 outcomes."""

    config = config or ExtractionConfig()
    audit = audit or ExtractionAudit()
    audit.articles_seen += 1
    if len(versions) < 3:
        audit.exclude(ExclusionReason.TOO_FEW_VERSIONS)
        return ()

    split_versions = [_sentences_for_version(version) for version in versions]
    examples: list[NewsEditsExample] = []
    for version_index in range(len(versions) - 2):
        audit.version_windows_seen += 1
        old_version, middle_version, future_version = versions[version_index : version_index + 3]
        old_sentences, middle_sentences, future_sentences = split_versions[
            version_index : version_index + 3
        ]
        if not old_sentences or not middle_sentences or not future_sentences:
            audit.exclude(ExclusionReason.EMPTY_VERSION)
            continue

        matcher = difflib.SequenceMatcher(
            a=[normalise_sentence(value) for value in old_sentences],
            b=[normalise_sentence(value) for value in middle_sentences],
            autojunk=False,
        )
        future_map = sentence_future_map(middle_sentences, future_sentences)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            audit.replacement_opcodes_seen += 1
            if (i2 - i1) != 1 or (j2 - j1) != 1:
                audit.exclude(ExclusionReason.AMBIGUOUS_REPLACEMENT)
                continue

            rejected = old_sentences[i1]
            selected = middle_sentences[j1]
            if not valid_sentence(
                rejected,
                min_chars=config.min_sentence_chars,
                max_chars=config.max_sentence_chars,
            ):
                audit.exclude(ExclusionReason.INVALID_REJECTED_SENTENCE)
                continue
            if not valid_sentence(
                selected,
                min_chars=config.min_sentence_chars,
                max_chars=config.max_sentence_chars,
            ):
                audit.exclude(ExclusionReason.INVALID_SELECTED_SENTENCE)
                continue
            if j1 not in future_map:
                audit.exclude(ExclusionReason.FUTURE_UNRESOLVED)
                continue

            similarity = difflib.SequenceMatcher(
                a=normalise_sentence(rejected),
                b=normalise_sentence(selected),
                autojunk=False,
            ).ratio()
            if similarity < config.min_edit_similarity:
                audit.exclude(ExclusionReason.SIMILARITY_BELOW_MIN)
                continue
            if similarity > config.max_edit_similarity:
                audit.exclude(ExclusionReason.SIMILARITY_ABOVE_MAX)
                continue

            before, after = surrounding_context(
                middle_sentences,
                j1,
                before=config.context_before,
                after=config.context_after,
            )
            lineage_id = middle_version.lineage_id
            triplet = RevisionTriplet(
                episode_id=(
                    f"{lineage_id}::{old_version.version_id}->{middle_version.version_id}::{j1}"
                ),
                lineage_id=lineage_id,
                v0_sentence=rejected,
                v1_sentence=selected,
                v2_sentence=future_map[j1],
            )
            examples.append(
                NewsEditsExample(
                    triplet=triplet,
                    source=middle_version.source,
                    article_id=middle_version.article_id,
                    v0_version_id=old_version.version_id,
                    v1_version_id=middle_version.version_id,
                    v2_version_id=future_version.version_id,
                    selected_sentence_index=j1,
                    context_before=before,
                    context_after=after,
                    sentence_position=j1 / max(1, len(middle_sentences) - 1),
                    edit_similarity=similarity,
                    lexical_jaccard=lexical_jaccard(rejected, selected),
                )
            )
            audit.accepted_examples += 1
    return tuple(examples)


def extract_from_database(
    connection: sqlite3.Connection,
    schema: ArticleSchema,
    *,
    config: ExtractionConfig | None = None,
    max_articles: int = 0,
    max_examples: int = 0,
    seed: int = 0,
    sources: Sequence[str] = (),
) -> ExtractionResult:
    """Extract examples from a full article-version table."""

    keys = sample_article_keys(
        connection,
        schema,
        max_articles=max_articles,
        seed=seed,
        sources=sources,
    )
    return _extract_version_stream(
        iter_article_versions(connection, schema, keys),
        config=config,
        max_examples=max_examples,
    )


def extract_from_split_database(
    connection: sqlite3.Connection,
    schema: SplitSentenceSchema,
    *,
    source_name: str,
    config: ExtractionConfig | None = None,
    max_articles: int = 0,
    max_examples: int = 0,
    seed: int = 0,
) -> ExtractionResult:
    """Extract examples from the official NewsEdits ``split_sentences`` table."""

    selected_ids = sample_split_article_ids(
        connection,
        schema,
        max_articles=max_articles,
        seed=seed,
    )
    return _extract_version_stream(
        iter_split_article_versions(
            connection,
            schema,
            selected_ids,
            source_name=source_name,
        ),
        config=config,
        max_examples=max_examples,
    )


def _extract_version_stream(
    version_stream: Iterable[tuple[tuple[str, str], tuple[ArticleVersion, ...]]],
    *,
    config: ExtractionConfig | None,
    max_examples: int,
) -> ExtractionResult:
    config = config or ExtractionConfig()
    audit = ExtractionAudit()
    examples: list[NewsEditsExample] = []
    seen_episode_ids: set[str] = set()

    for _key, versions in version_stream:
        for example in extract_article_examples(versions, config=config, audit=audit):
            if example.triplet.episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(example.triplet.episode_id)
            examples.append(example)
            if max_examples > 0 and len(examples) >= max_examples:
                audit.finalize(examples)
                return ExtractionResult(examples=tuple(examples), audit=audit)

    audit.finalize(examples)
    return ExtractionResult(examples=tuple(examples), audit=audit)


def _sentences_for_version(version: ArticleVersion) -> list[str]:
    """Use official sentence rows when available; split only full article text."""

    if version.sentences is not None:
        return list(version.sentences)
    return sentence_split(version.text)
