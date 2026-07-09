"""Canonical episode construction for preference-to-future experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class RevisionTriplet:
    """Three observed states from one sentence lineage.

    ``v0_sentence`` is the sentence replaced at the decision boundary.
    ``v1_sentence`` is the replacement retained by the editor.
    ``v2_sentence`` is the next observed state of that retained branch.
    """

    episode_id: str
    lineage_id: str
    v0_sentence: str
    v1_sentence: str
    v2_sentence: str

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "lineage_id",
            "v0_sentence",
            "v1_sentence",
            "v2_sentence",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")

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

    @property
    def selected_candidate(self) -> str:
        """Return the candidate retained at the decision boundary."""

        return self.candidate_a if self.selected_index == 0 else self.candidate_b

    @property
    def rejected_candidate(self) -> str:
        """Return the candidate removed at the decision boundary."""

        return self.candidate_b if self.selected_index == 0 else self.candidate_a


def build_preference_episode(triplet: RevisionTriplet, *, seed: int) -> PreferenceEpisode:
    """Build a deterministic randomised pair from one revision triplet.

    The selected candidate is always V1 and the rejected candidate is always V0. Only their
    presentation order changes. The future outcome is defined by whether V1 changes in V2, so it
    remains attached to the selected branch rather than to candidate position.
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
        future_revised=_normalise_text(triplet.v2_sentence)
        != _normalise_text(triplet.v1_sentence),
    )


def _selected_goes_in_slot_a(episode_id: str, *, seed: int) -> bool:
    """Return a stable pseudo-random orientation without relying on process hash state."""

    digest = sha256(f"{seed}\0{episode_id}".encode("utf-8")).digest()
    return bool(digest[0] & 1)


def _normalise_text(text: str) -> str:
    """Normalise inconsequential whitespace while preserving textual revisions."""

    return " ".join(text.split())
