import json

import pytest

from preference_futures.episodes import (
    EPISODE_SCHEMA_VERSION,
    PreferenceEpisode,
    RevisionTriplet,
    build_preference_episode,
)


def test_randomised_candidate_order_preserves_preference_and_future_lineage() -> None:
    triplet = RevisionTriplet(
        episode_id="article-7:sentence-3:v0-v1-v2",
        lineage_id="article-7",
        v0_sentence="The committee may meet on Monday.",
        v1_sentence="The committee will meet on Monday.",
        v2_sentence="The committee will meet on Tuesday.",
    )

    episodes = [build_preference_episode(triplet, seed=seed) for seed in range(256)]

    assert {episode.selected_index for episode in episodes} == {0, 1}
    assert all(episode.selected_candidate == triplet.v1_sentence for episode in episodes)
    assert all(episode.rejected_candidate == triplet.v0_sentence for episode in episodes)
    assert all(episode.future_revised is True for episode in episodes)

    repeated = build_preference_episode(triplet, seed=17)
    assert repeated == build_preference_episode(triplet, seed=17)


def test_whitespace_only_v1_to_v2_change_is_stable() -> None:
    triplet = RevisionTriplet(
        episode_id="article-8:sentence-4:v0-v1-v2",
        lineage_id="article-8",
        v0_sentence="The vote could happen today.",
        v1_sentence="The vote will happen today.",
        v2_sentence="  The vote   will happen today.\n",
    )

    episode = build_preference_episode(triplet, seed=3)

    assert episode.future_revised is False
    assert episode.selected_candidate == triplet.v1_sentence


def test_missing_v2_continuation_is_revised_or_removed() -> None:
    triplet = RevisionTriplet(
        episode_id="article-8:sentence-5:v0-v1-v2",
        lineage_id="article-8",
        v0_sentence="The vote could happen tomorrow.",
        v1_sentence="The vote will happen tomorrow.",
        v2_sentence=None,
    )

    assert build_preference_episode(triplet, seed=3).future_revised is True


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("episode_id", ""),
        ("lineage_id", "   "),
        ("v0_sentence", ""),
        ("v1_sentence", "\n"),
        ("v2_sentence", ""),
    ],
)
def test_revision_triplet_rejects_empty_fields(field_name: str, value: str) -> None:
    values = {
        "episode_id": "episode-1",
        "lineage_id": "lineage-1",
        "v0_sentence": "Earlier wording.",
        "v1_sentence": "Replacement wording.",
        "v2_sentence": "Replacement wording.",
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        RevisionTriplet(**values)


def test_revision_triplet_requires_a_real_v0_to_v1_replacement() -> None:
    with pytest.raises(ValueError, match="real replacement"):
        RevisionTriplet(
            episode_id="episode-1",
            lineage_id="lineage-1",
            v0_sentence="The same sentence.",
            v1_sentence="  The same   sentence.\n",
            v2_sentence="The same sentence.",
        )


def test_episode_record_round_trips_through_json() -> None:
    original = PreferenceEpisode(
        episode_id="episode-9",
        lineage_id="article-9",
        candidate_a="Rejected wording.",
        candidate_b="Selected wording.",
        selected_index=1,
        future_revised=False,
    )

    encoded = json.dumps(original.to_record())
    restored = PreferenceEpisode.from_record(json.loads(encoded))

    assert restored == original
    assert restored.to_record()["schema_version"] == EPISODE_SCHEMA_VERSION


def test_episode_rejects_invalid_selected_index() -> None:
    with pytest.raises(ValueError, match="selected_index"):
        PreferenceEpisode(
            episode_id="episode-1",
            lineage_id="lineage-1",
            candidate_a="Candidate A.",
            candidate_b="Candidate B.",
            selected_index=2,
            future_revised=False,
        )


def test_episode_record_rejects_unknown_schema_version() -> None:
    record = {
        "schema_version": EPISODE_SCHEMA_VERSION + 1,
        "episode_id": "episode-1",
        "lineage_id": "lineage-1",
        "candidate_a": "Candidate A.",
        "candidate_b": "Candidate B.",
        "selected_index": 0,
        "future_revised": False,
    }

    with pytest.raises(ValueError, match="unsupported schema_version"):
        PreferenceEpisode.from_record(record)
