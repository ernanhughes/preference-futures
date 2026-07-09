"""Shared helpers for fixed-budget representation training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.corpora.common import CORPUS_NAMES

TRAINING_CONTRACT_SCHEMA_VERSION = 1
TRAINING_RUN_SCHEMA_VERSION = 1
TRAINING_VERIFICATION_SCHEMA_VERSION = 1
TRAINED_REGIMES = tuple(CORPUS_NAMES)
CLASSIFICATION_REGIMES = (
    "pair_exposure",
    "temporal_direction",
    "random_label",
    "shuffled_preference",
    "authentic_preference",
)
LANGUAGE_ADAPTATION_REGIME = "language_adaptation"


def sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash file names and contents in a directory using a stable relative order."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"directory does not exist: {resolved}")
    digest = hashlib.sha256()
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory contains no files: {resolved}")
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {resolved}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: expected an object")
            records.append(value)
    return records


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_int_selection(value: str, *, upper_bound: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(upper_bound))
    selected: set[int] = set()
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            left, right = text.split("-", maxsplit=1)
            start = int(left)
            end = int(right)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(text))
    if not selected or min(selected) < 0 or max(selected) >= upper_bound:
        raise ValueError(f"selection must contain values from 0 to {upper_bound - 1}")
    return tuple(sorted(selected))


def parse_regime_selection(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return TRAINED_REGIMES
    requested = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(requested).difference(TRAINED_REGIMES)
    if not requested or unknown:
        raise ValueError(f"unknown or empty regime selection: {sorted(unknown)}")
    return requested


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def require_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be an array")
    return value
