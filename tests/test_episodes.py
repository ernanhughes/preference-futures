from preference_futures.episodes import RevisionTriplet, build_preference_episode


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
