from __future__ import annotations

from preference_futures.diagnostics.labels import (
    build_future_label_integrity_report,
    categorise_normalisation_mismatch,
)
from preference_futures.diagnostics.preference import (
    balanced_record_subset,
    build_preference_learnability_summary,
    classification_diagnostic,
)
from preference_futures.episodes import RevisionTriplet
from preference_futures.newsedits.models import NewsEditsExample


def _records(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "source_id": f"episode-{index:03d}",
            "target": index % 2,
        }
        for index in range(count)
    ]


def _example(*, episode_id: str, v1: str, v2: str | None) -> NewsEditsExample:
    return NewsEditsExample(
        triplet=RevisionTriplet(
            episode_id=episode_id,
            lineage_id=f"lineage-{episode_id}",
            v0_sentence="This is the older sentence with enough words.",
            v1_sentence=v1,
            v2_sentence=v2,
        ),
        source="fixture",
        article_id=episode_id,
        v0_version_id="v0",
        v1_version_id="v1",
        v2_version_id="v2",
        selected_sentence_index=0,
        context_before="Before context.",
        context_after="After context.",
        sentence_position=0.5,
        edit_similarity=0.8,
        lexical_jaccard=0.7,
    )


def test_balanced_subset_is_deterministic_and_balanced() -> None:
    records = _records()

    first = balanced_record_subset(records, size=10, seed=17)
    second = balanced_record_subset(records, size=10, seed=17)

    assert first == second
    assert sum(int(record["target"]) == 0 for record in first) == 5
    assert sum(int(record["target"]) == 1 for record in first) == 5


def test_classification_diagnostic_distinguishes_null_and_learned() -> None:
    records = _records(1000)

    null = classification_diagnostic(
        {"accuracy": 0.5, "mean_loss": 0.6932},
        records,
    )
    learned = classification_diagnostic(
        {"accuracy": 0.6, "mean_loss": 0.65},
        records,
    )

    assert null["source_task_status"] == "null_like"
    assert learned["source_task_status"] == "learned_above_prior"


def test_summary_prioritises_failed_memorization_gate() -> None:
    reports = [
        {
            "condition_name": "memorize-256",
            "condition_kind": "tiny_set_memorization",
            "update_steps": 5000,
            "train_evaluation": {"accuracy": 0.7},
            "validation": {"accuracy": 0.5, "mean_loss": 0.693},
            "validation_diagnostic": {
                "source_task_status": "null_like",
                "accuracy_interval_95": [0.45, 0.55],
            },
        },
        {
            "condition_name": "budget-2400",
            "condition_kind": "full_train_learning_curve",
            "update_steps": 2400,
            "train_evaluation": {"accuracy": 0.51},
            "validation": {"accuracy": 0.5, "mean_loss": 0.693},
            "validation_diagnostic": {
                "source_task_status": "null_like",
                "accuracy_interval_95": [0.45, 0.55],
            },
        },
    ]

    summary = build_preference_learnability_summary(
        contract={"contract_sha256": "contract"},
        fold=0,
        reports=reports,
        surface_baselines={},
    )

    assert summary["outcome"] == "tiny_set_memorization_failed"
    assert summary["gates"]["tiny_set_memorization_passed"] is False


def test_label_audit_counts_case_and_quote_mismatches() -> None:
    examples = [
        _example(
            episode_id="case",
            v1="The Editor Retained This Complete Sentence.",
            v2="the editor retained this complete sentence.",
        ),
        _example(
            episode_id="quotes",
            v1="The editor called it “a complete sentence” today.",
            v2='The editor called it "a complete sentence" today.',
        ),
        _example(
            episode_id="real-change",
            v1="The editor retained this complete sentence today.",
            v2="The editor replaced this sentence tomorrow.",
        ),
        _example(
            episode_id="missing",
            v1="The editor retained another complete sentence today.",
            v2=None,
        ),
    ]

    report = build_future_label_integrity_report(examples, sample_limit=10)

    assert report["counts"]["episodes"] == 4
    assert report["counts"]["normalization_mismatch_labels"] == 2
    assert report["mismatch_categories"] == {
        "case_only": 1,
        "quote_style_only": 1,
    }
    assert report["counts"]["current_future_revised"] == 4
    assert report["counts"]["corrected_normalized_future_revised"] == 2


def test_mismatch_category_handles_combined_case_and_quotes() -> None:
    assert (
        categorise_normalisation_mismatch(
            "The editor called it “Final”.",
            'the editor called it "final".',
        )
        == "case_and_quote_style"
    )
