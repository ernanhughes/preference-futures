"""Canonical episode construction for preference-to-future experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Self

EPISODE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RevisionTriplet:
    """Three observed states from one sentence lineage.

    ``v0_sentence`` is the sentence replaced at the decision boundary.
    ``v1_sentence`` is the replacement retained by the editor.
    ``v2_sentence`` is its unambiguous next state, or ``None`` when the selected
    sentence was revised/removed without a one-to-one continuation.
    """

    episode_id: str
    lineage_id: str
    v0_sentence: str
    v1_sentence: str
    v2_sentence: str | None

    def __post_init__(self) -> None:
        for field_name in ("episode_id", "lineage_id", "v0_sentence", "v1_sentence"):
            _validate_non_empty_text(field_name, getattr(self, field_name))

        if self.v2_sentence is not None:
            _validate_non_empty_text("v2_sentence", self.v2_sentence)

        if _normalise_text(self.v0_sentence) == _normalise_text(self.v1_sentence):
            raise ValueError("v0_sentence and v1_sentence must describe a real replacement")


@dataclass(frozen=True, slots=True)
class PreferenceEpisode:
    """A randomised preference pair with a future label attached to the selected branch."""

    episode_id: str
    lineage_id: str
    candidate_a: str
    candidate_b: str
    selected_index: int
    future_revised: bool

    def __post_init__(self) -> None:
        for field_name in ("episode_id", "lineage_id", "candidate_a", "candidate_b"):
            _validate_non_empty_text(field_name, getattr(self, field_name))

        if _normalise_text(self.candidate_a) == _normalise_text(self.candidate_b):
            raise ValueError("candidate_a and candidate_b must be distinct")

        if type(self.selected_index) is not int or self.selected_index not in (0, 1):
            raise ValueError("selected_index must be either 0 or 1")

        if type(self.future_revised) is not bool:
            raise TypeError("future_revised must be a bool")

    @property
    def selected_candidate(self) -> str:
        """Return the candidate retained at the decision boundary."""

        return self.candidate_a if self.selected_index == 0 else self.candidate_b

    @property
    def rejected_candidate(self) -> str:
        """Return the candidate removed at the decision boundary."""

        return self.candidate_b if self.selected_index == 0 else self.candidate_a

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible, versioned dataset record."""

        return {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "lineage_id": self.lineage_id,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "selected_index": self.selected_index,
            "future_revised": self.future_revised,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Self:
        """Reconstruct and validate an episode from a dataset record."""

        schema_version = record.get("schema_version")
        if type(schema_version) is not int:
            raise TypeError("schema_version must be an int")
        if schema_version != EPISODE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {schema_version}; "
                f"expected {EPISODE_SCHEMA_VERSION}"
            )

        selected_index = record.get("selected_index")
        if type(selected_index) is not int:
            raise TypeError("selected_index must be an int")

        future_revised = record.get("future_revised")
        if type(future_revised) is not bool:
            raise TypeError("future_revised must be a bool")

        return cls(
            episode_id=_record_text(record, "episode_id"),
            lineage_id=_record_text(record, "lineage_id"),
            candidate_a=_record_text(record, "candidate_a"),
            candidate_b=_record_text(record, "candidate_b"),
            selected_index=selected_index,
            future_revised=future_revised,
        )


def build_preference_episode(triplet: RevisionTriplet, *, seed: int) -> PreferenceEpisode:
    """Build a deterministic randomised pair from one revision triplet.

    The selected candidate is always V1 and the rejected candidate is always V0.
    Only their presentation order changes. A missing one-to-one V2 continuation
    counts as revised/removed.
    """

    selected_is_a = _selected_goes_in_slot_a(triplet.episode_id, seed=seed)
    if selected_is_a:
        candidate_a = triplet.v1_sentence
        candidate_b = triplet.v0_sentence
        selected_index = 0
    else:
        candidate_a = triplet.v0_sentence
        candidate_b = triplet.v1_sentence
        selected_index = 1

    return PreferenceEpisode(
        episode_id=triplet.episode_id,
        lineage_id=triplet.lineage_id,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        selected_index=selected_index,
        future_revised=(
            triplet.v2_sentence is None
            or _normalise_text(triplet.v2_sentence) != _normalise_text(triplet.v1_sentence)
        ),
    )


def _selected_goes_in_slot_a(episode_id: str, *, seed: int) -> bool:
    """Return a stable pseudo-random orientation without process hash state."""

    digest = sha256(f"{seed}\0{episode_id}".encode("utf-8")).digest()
    return bool(digest[0] & 1)


def _record_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a str")
    return value


def _validate_non_empty_text(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _normalise_text(text: str) -> str:
    """Normalise inconsequential whitespace while preserving textual revisions."""

    return " ".join(text.split())
