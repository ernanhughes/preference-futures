from __future__ import annotations

import pytest

from preference_futures.editorial_mrq.matched_common import (
    ARMS,
    comparison_passed,
    parse_arms,
)


def test_parse_matched_control_arms() -> None:
    assert parse_arms("all") == ARMS
    assert parse_arms("pca_generic_unoriented,extended_generic_choice_aware") == (
        "pca_generic_unoriented",
        "extended_generic_choice_aware",
    )
    with pytest.raises(ValueError, match="unknown or empty"):
        parse_arms("unknown")


def test_matched_control_comparison_requires_interval_below_zero() -> None:
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
    assert not comparison_passed(
        {
            "mean_log_loss_difference": 0.001,
            "confidence_interval_95": [-0.01, -0.001],
        }
    )
