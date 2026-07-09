"""Shared constants and helpers for Step 5 frozen representation extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from preference_futures.selection.diagnostics import ALL_ARMS
from preference_futures.training.common import canonical_json_sha256

REPRESENTATION_CONTRACT_SCHEMA_VERSION = 1
REPRESENTATION_RUN_SCHEMA_VERSION = 1
REPRESENTATION_VERIFICATION_SCHEMA_VERSION = 1
PARTITIONS = ("train", "validation", "test")
FORBIDDEN_ROW_KEYS = frozenset(
    {
        "future_revised",
        "future_label",
        "selected_index",
        "selected_candidate",
        "v2_sentence",
        "v2_text",
        "v2_identifier",
        "later_outcome",
    }
)


def validate_embedded_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    label: str,
) -> None:
    expected = str(value.get(hash_field, ""))
    payload = dict(value)
    payload.pop(hash_field, None)
    observed = canonical_json_sha256(payload)
    if not expected or observed != expected:
        raise ValueError(f"{label} canonical hash is missing or invalid")


def parse_arm_selection(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(ALL_ARMS)
    requested = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(requested).difference(ALL_ARMS)
    if not requested or unknown:
        raise ValueError(f"unknown or empty arm selection: {sorted(unknown)}")
    return requested
