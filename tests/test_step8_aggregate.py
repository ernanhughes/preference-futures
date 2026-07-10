from __future__ import annotations

from preference_futures.editorial_mrq.aggregate import paired_comparison, wilson_interval


def test_paired_comparison_prefers_better_mrq_probabilities() -> None:
    linear = {
        "one": (1, 0.55, "lineage-a"),
        "two": (0, 0.45, "lineage-b"),
        "three": (1, 0.40, "lineage-c"),
        "four": (0, 0.60, "lineage-d"),
    }
    mrq = {
        "one": (1, 0.70, "lineage-a"),
        "two": (0, 0.30, "lineage-b"),
        "three": (1, 0.60, "lineage-c"),
        "four": (0, 0.40, "lineage-d"),
    }

    result = paired_comparison(linear, mrq)

    assert result["records"] == 4
    assert result["mrq_minus_linear_log_loss"] < 0.0
    assert result["mrq_better_log_loss"] is True
    assert result["mrq_minus_linear_accuracy"] > 0.0


def test_paired_comparison_rejects_different_episode_sets() -> None:
    linear = {"one": (1, 0.5, "lineage-a")}
    mrq = {"two": (1, 0.5, "lineage-a")}

    try:
        paired_comparison(linear, mrq)
    except ValueError as exc:
        assert "identical episodes" in str(exc)
    else:
        raise AssertionError("different episode sets should be rejected")


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(60, 100)

    assert lower < 0.6 < upper
    assert lower > 0.5
