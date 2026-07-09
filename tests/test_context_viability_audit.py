from __future__ import annotations

from preference_futures.audit import (
    build_context_viability_report,
    render_context_viability_markdown,
)


def _record(
    *,
    episode_id: str,
    lineage_id: str,
    candidate_a: str,
    candidate_b: str,
    selected_index: int,
    future_revised: bool,
    v1_version_id: str,
    context_before: str = "Before sentence.",
    context_after: str = "After sentence.",
    sentence_position: float = 0.5,
    edit_similarity: float = 0.9,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "lineage_id": lineage_id,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "selected_index": selected_index,
        "future_revised": future_revised,
        "v0_version_id": str(int(v1_version_id) - 1),
        "v1_version_id": v1_version_id,
        "v2_version_id": str(int(v1_version_id) + 1),
        "selected_sentence_index": 2,
        "context_before": context_before,
        "context_after": context_after,
        "sentence_position": sentence_position,
        "edit_similarity": edit_similarity,
        "lexical_jaccard": 0.8,
    }


def test_report_detects_pair_reversal_and_boundary_artifact() -> None:
    records = [
        _record(
            episode_id="one",
            lineage_id="nyt::1",
            candidate_a="The newer sentence.",
            candidate_b="The older sentence.",
            selected_index=0,
            future_revised=True,
            v1_version_id="1",
            context_before="Mr.",
        ),
        _record(
            episode_id="two",
            lineage_id="nyt::1",
            candidate_a="The newer sentence.",
            candidate_b="The older sentence.",
            selected_index=1,
            future_revised=False,
            v1_version_id="2",
        ),
        _record(
            episode_id="three",
            lineage_id="nyt::2",
            candidate_a="Another selected sentence.",
            candidate_b="Another rejected sentence.",
            selected_index=0,
            future_revised=False,
            v1_version_id="11",
            context_after="This article has been revised to reflect a correction.",
        ),
        _record(
            episode_id="four",
            lineage_id="nyt::3",
            candidate_a="A fourth selected sentence.",
            candidate_b="A fourth rejected sentence.",
            selected_index=1,
            future_revised=True,
            v1_version_id="12",
        ),
    ]

    report = build_context_viability_report(records)

    assert report["dataset"]["episodes"] == 4
    assert report["dataset"]["lineages"] == 3
    assert report["target_balance"]["future_revised_rate"] == 0.5
    assert report["candidate_orientation"]["selected_b_rate"] == 0.5
    assert report["context"]["source_boundary_artifacts"] == 1
    assert report["context"]["boilerplate_or_credit_records"] == 1
    assert report["pair_reuse"]["reversal_pair_groups"] == 1
    assert report["pair_reuse"]["same_lineage_reversal_groups"] == 1
    assert report["pair_reuse"]["episodes_in_reversal_groups"] == 2

    markdown = render_context_viability_markdown(report)
    assert "# Context Viability Audit" in markdown
    assert "Reversal pair groups | 1" in markdown
    assert "at least 10000 episodes | FAIL" in markdown
