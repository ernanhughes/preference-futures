"""Shared constants and helpers for Step 6 identical future probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from preference_futures.representations.common import parse_arm_selection, validate_embedded_hash

PROBE_CONTRACT_SCHEMA_VERSION = 1
PROBE_RUN_SCHEMA_VERSION = 1
PROBE_VERIFICATION_SCHEMA_VERSION = 1
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SELECTION_TOLERANCE = 1e-12
PROBABILITY_EPSILON = 1e-7
STANDARDISATION_EPSILON = 1e-6


def select_l2_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = SELECTION_TOLERANCE,
) -> Mapping[str, Any]:
    """Select lowest validation log loss, breaking numerical ties toward stronger L2."""

    if not candidates:
        raise ValueError("probe candidate list is empty")
    best_loss = min(float(candidate["validation"]["log_loss"]) for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if float(candidate["validation"]["log_loss"]) <= best_loss + tolerance
    ]
    return max(tied, key=lambda candidate: float(candidate["l2_lambda"]))


__all__ = [
    "L2_GRID",
    "PROBABILITY_EPSILON",
    "PROBE_CONTRACT_SCHEMA_VERSION",
    "PROBE_RUN_SCHEMA_VERSION",
    "PROBE_VERIFICATION_SCHEMA_VERSION",
    "SELECTION_TOLERANCE",
    "STANDARDISATION_EPSILON",
    "parse_arm_selection",
    "select_l2_candidate",
    "validate_embedded_hash",
]
