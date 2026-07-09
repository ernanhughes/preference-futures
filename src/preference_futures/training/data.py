"""Materialize compact Step 2 corpus records into Step 3 model inputs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preference_futures.training.common import load_jsonl


@dataclass(frozen=True, slots=True)
class SourceStore:
    episodes: dict[str, dict[str, Any]]
    temporal_pairs: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ClassificationExample:
    source_id: str
    text: str
    target: int


@dataclass(frozen=True, slots=True)
class MaskedLanguageExample:
    source_id: str
    words: tuple[str, ...]
    mask_word_indices: tuple[int, ...]


def load_source_store(episodes_path: Path, temporal_pairs_path: Path) -> SourceStore:
    episodes = _index_records(load_jsonl(episodes_path), "episode_id")
    temporal_pairs = _index_records(load_jsonl(temporal_pairs_path), "temporal_pair_id")
    return SourceStore(episodes=episodes, temporal_pairs=temporal_pairs)


def materialize_record(
    record: dict[str, Any], store: SourceStore
) -> ClassificationExample | MaskedLanguageExample:
    corpus = str(record.get("corpus", ""))
    source_id = str(record.get("source_id", ""))
    if not source_id:
        raise ValueError("source-task record has no source_id")

    if corpus == "language_adaptation":
        episode = _required_source(store.episodes, source_id, "episode")
        words = tuple(serialise_episode(episode).split())
        raw_indices = record.get("mask_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError(f"language-adaptation record has no mask indices: {source_id}")
        indices = tuple(sorted({int(index) for index in raw_indices}))
        if min(indices) < 0 or max(indices) >= len(words):
            raise ValueError(f"language-adaptation mask index is out of range: {source_id}")
        return MaskedLanguageExample(
            source_id=source_id,
            words=words,
            mask_word_indices=indices,
        )

    target = record.get("target")
    if type(target) is not int or target not in (0, 1):
        raise ValueError(f"classification record has invalid target: {source_id}")

    if corpus == "temporal_direction":
        pair = _required_source(store.temporal_pairs, source_id, "temporal pair")
        earlier = str(pair["earlier_text"])
        later = str(pair["later_text"])
        if target == 1:
            candidate_a, candidate_b = earlier, later
        else:
            candidate_a, candidate_b = later, earlier
        text = serialise_fields(
            context_before=str(pair.get("context_before", "")),
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            context_after=str(pair.get("context_after", "")),
        )
        return ClassificationExample(source_id=source_id, text=text, target=target)

    episode = _required_source(store.episodes, source_id, "episode")
    candidate_b = str(episode["candidate_b"])
    if corpus == "pair_exposure" and target == 0:
        donor_id = str(record.get("candidate_b_source_episode_id", ""))
        donor = _required_source(store.episodes, donor_id, "pair donor episode")
        if str(donor["lineage_id"]) == str(episode["lineage_id"]):
            raise ValueError(f"pair-exposure donor does not cross lineages: {source_id}")
        candidate_b = str(donor["candidate_b"])

    text = serialise_episode(episode, candidate_b=candidate_b)
    return ClassificationExample(source_id=source_id, text=text, target=target)


def serialise_episode(episode: dict[str, Any], *, candidate_b: str | None = None) -> str:
    return serialise_fields(
        context_before=str(episode.get("context_before", "")),
        candidate_a=str(episode["candidate_a"]),
        candidate_b=str(episode["candidate_b"] if candidate_b is None else candidate_b),
        context_after=str(episode.get("context_after", "")),
    )


def serialise_fields(
    *, context_before: str, candidate_a: str, candidate_b: str, context_after: str
) -> str:
    return "\n".join(
        (
            "Context before:",
            context_before,
            "Candidate A:",
            candidate_a,
            "Candidate B:",
            candidate_b,
            "Context after:",
            context_after,
        )
    )


def deterministic_training_batches(
    record_count: int,
    *,
    batch_size: int,
    update_steps: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Return fixed-size batches, cycling deterministically when the budget exceeds one pass."""

    if record_count < 1 or batch_size < 1 or update_steps < 1:
        raise ValueError("record_count, batch_size and update_steps must be positive")
    batches: list[tuple[int, ...]] = []
    epoch = 0
    order: list[int] = []
    cursor = 0
    while len(batches) < update_steps:
        if cursor >= len(order):
            order = list(range(record_count))
            random.Random(seed + epoch).shuffle(order)
            cursor = 0
            epoch += 1
        batch: list[int] = []
        while len(batch) < batch_size:
            if cursor >= len(order):
                order = list(range(record_count))
                random.Random(seed + epoch).shuffle(order)
                cursor = 0
                epoch += 1
            take = min(batch_size - len(batch), len(order) - cursor)
            batch.extend(order[cursor : cursor + take])
            cursor += take
        batches.append(tuple(batch))
    return batches


def sequential_validation_batches(record_count: int, *, batch_size: int) -> list[tuple[int, ...]]:
    if record_count < 1 or batch_size < 1:
        raise ValueError("record_count and batch_size must be positive")
    return [
        tuple(range(start, min(start + batch_size, record_count)))
        for start in range(0, record_count, batch_size)
    ]


def _index_records(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = str(record.get(key, ""))
        if not identifier or identifier in indexed:
            raise ValueError(f"missing or duplicate {key}: {identifier!r}")
        indexed[identifier] = record
    if not indexed:
        raise ValueError(f"source contains no records keyed by {key}")
    return indexed


def _required_source(
    sources: dict[str, dict[str, Any]], identifier: str, label: str
) -> dict[str, Any]:
    try:
        return sources[identifier]
    except KeyError as exc:
        raise ValueError(f"missing {label}: {identifier}") from exc
