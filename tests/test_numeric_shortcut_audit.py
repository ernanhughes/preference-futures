from __future__ import annotations

from preference_futures.audit.numeric import (
    build_numeric_shortcut_report,
    classify_numeric_episode,
    render_numeric_shortcut_markdown,
)


def _record(
    *,
    episode_id: str,
    lineage_id: str,
    candidate_a: str,
    candidate_b: str,
    future_revised: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "lineage_id": lineage_id,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "selected_index": 0,
        "future_revised": future_revised,
        "v0_version_id": "1",
        "v1_version_id": "2",
        "v2_version_id": "3",
        "selected_sentence_index": 0,
        "context_before": "",
        "context_after": "",
        "sentence_position": 0.0,
        "edit_similarity": 0.9,
        "lexical_jaccard": 0.8,
    }


def test_classifies_number_only_and_casualty_update() -> None:
    record = _record(
        episode_id="one",
        lineage_id="nyt::1",
        candidate_a="At least 50 people were killed in the attack.",
        candidate_b="At least 40 people were killed in the attack.",
        future_revised=True,
    )

    result = classify_numeric_episode(record)

    assert result["contains_number"] is True
    assert result["number_changed"] is True
    assert result["number_only_edit"] is True
    assert result["number_dominant_edit"] is True
    assert result["casualty_count_update"] is True
    assert result["numbers_a"] == ["50"]
    assert result["numbers_b"] == ["40"]


def test_report_marks_repeated_numeric_trajectory_and_complement_rate() -> None:
    records = [
        _record(
            episode_id="one",
            lineage_id="nyt::1",
            candidate_a="At least 50 people were killed.",
            candidate_b="At least 40 people were killed.",
            future_revised=True,
        ),
        _record(
            episode_id="two",
            lineage_id="nyt::1",
            candidate_a="At least 40 people were killed.",
            candidate_b="At least 30 people were killed.",
            future_revised=True,
        ),
        _record(
            episode_id="three",
            lineage_id="nyt::2",
            candidate_a="The committee may meet Monday.",
            candidate_b="The committee will meet Monday.",
            future_revised=False,
        ),
        _record(
            episode_id="four",
            lineage_id="nyt::3",
            candidate_a="Shares rose 5 percent.",
            candidate_b="Shares rose 7 percent.",
            future_revised=False,
        ),
    ]

    report, flags = build_numeric_shortcut_report(records)

    changed = report["categories"]["number_changed"]
    assert changed["episodes"] == 3
    assert changed["future_revised"] == 2
    assert changed["future_revised_rate"] == 2 / 3
    assert changed["future_revised_rate_without_category"] == 0.0
    assert changed["risk_ratio_vs_complement"] is None
    assert report["trajectories"]["repeated_numeric_lineages"] == 1
    assert report["trajectories"]["repeated_casualty_lineages"] == 1
    assert sum(item["repeated_numeric_trajectory"] for item in flags) == 2
    assert sum(item["repeated_casualty_trajectory"] for item in flags) == 2
    assert any("risk ratio is undefined" in item for item in report["interpretation"])

    markdown = render_numeric_shortcut_markdown(report)
    assert "# Numeric Shortcut Audit" in markdown
    assert "number changed" in markdown
    assert "Required controls" in markdown
