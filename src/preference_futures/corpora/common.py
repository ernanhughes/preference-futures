"""Shared helpers for Step 2 corpus construction and verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CORPUS_SCHEMA_VERSION = 1
CORPUS_NAMES = (
    "language_adaptation",
    "pair_exposure",
    "temporal_direction",
    "random_label",
    "shuffled_preference",
    "authentic_preference",
)
FORBIDDEN_FUTURE_KEYS = {
    "future_revised",
    "future_stable",
    "v2_sentence",
    "v2_version_id",
    "future_label",
    "future_outcome",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL on line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(value)
    return records


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def source_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
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


def hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def string_set(value: Any, name: str) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} lineages must be an array")
    return {str(item) for item in value}


def positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_episodes(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    required = {
        "episode_id",
        "lineage_id",
        "candidate_a",
        "candidate_b",
        "selected_index",
        "context_before",
        "context_after",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in episodes:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"episode missing fields: {sorted(missing)}")
        episode_id = str(record["episode_id"])
        if episode_id in by_id:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        if type(record["selected_index"]) is not int or record["selected_index"] not in (0, 1):
            raise ValueError(f"invalid selected_index for {episode_id}")
        by_id[episode_id] = record
    return by_id


def validate_temporal_pairs(
    temporal_pairs: Sequence[Mapping[str, Any]], evaluation_lineages: set[str]
) -> dict[str, Mapping[str, Any]]:
    if not temporal_pairs:
        raise ValueError("temporal_pairs must not be empty")
    required = {"temporal_pair_id", "lineage_id", "earlier_text", "later_text"}
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in temporal_pairs:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"temporal pair missing fields: {sorted(missing)}")
        pair_id = str(record["temporal_pair_id"])
        lineage_id = str(record["lineage_id"])
        if pair_id in by_id:
            raise ValueError(f"duplicate temporal_pair_id: {pair_id}")
        if lineage_id in evaluation_lineages:
            raise ValueError(f"temporal pair overlaps evaluation lineage: {lineage_id}")
        by_id[pair_id] = record
    return by_id


def assert_partition(
    train: set[str], validation: set[str], test: set[str], all_lineages: set[str]
) -> None:
    if train & validation or train & test or validation & test:
        raise ValueError("lineage leakage detected in fold document")
    if train | validation | test != all_lineages:
        raise ValueError("fold document does not cover every evaluation lineage")
