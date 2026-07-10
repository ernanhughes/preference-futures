from __future__ import annotations

import pytest

from preference_futures.editorial_mrq.xgboost_combined import (
    ARMS,
    comparison_passed,
    parse_arms,
    pooled_metrics,
    shuffled_replicate_from_arm,
)


def test_parse_xgboost_arms() -> None:
    assert parse_arms("all") == ARMS
    assert parse_arms("xgb_generic_all,xgb_authentic_mrq_only") == (
        "xgb_generic_all",
        "xgb_authentic_mrq_only",
    )
    with pytest.raises(ValueError, match="unknown or empty"):
        parse_arms("unknown")


def test_parse_shuffled_replicate() -> None:
    assert shuffled_replicate_from_arm("xgb_generic_plus_shuffled_mrq_r00") == 0
    assert shuffled_replicate_from_arm("xgb_generic_plus_shuffled_mrq_r04") == 4
    with pytest.raises(ValueError, match="not a shuffled"):
        shuffled_replicate_from_arm("xgb_generic_all")
    with pytest.raises(ValueError, match="invalid"):
        shuffled_replicate_from_arm("xgb_generic_plus_shuffled_mrq_r05")


def test_comparison_gate_requires_negative_upper_bound() -> None:
    assert comparison_passed(
        {
            "mean_log_loss_difference": -0.01,
            "confidence_interval_95": [-0.02, -0.001],
        }
    )
    assert not comparison_passed(
        {
            "mean_log_loss_difference": -0.01,
            "confidence_interval_95": [-0.02, 0.001],
        }
    )


def test_pooled_metrics_uses_unique_predictions() -> None:
    metrics = pooled_metrics(
        {
            "a": (1, 0.8, "l1"),
            "b": (0, 0.2, "l2"),
            "c": (1, 0.7, "l3"),
        }
    )
    assert metrics["records"] == 3
    assert metrics["accuracy"] == 1.0
    assert metrics["log_loss"] < 0.4
