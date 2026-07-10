from __future__ import annotations

import pytest

from preference_futures.editorial_mrq.transfer import (
    _parse_arms,
    lineage_bootstrap_interval,
    paired_transfer_comparison,
)


def test_parse_transfer_arms() -> None:
    assert _parse_arms("mrq_blind,generic_unoriented") == (
        "mrq_blind",
        "generic_unoriented",
    )
    with pytest.raises(ValueError, match="unknown or empty"):
        _parse_arms("unknown")


def test_lineage_bootstrap_is_deterministic() -> None:
    values = {"a": [-0.1, -0.2], "b": [0.05], "c": [-0.03, -0.04]}
    first = lineage_bootstrap_interval(values, seed=17, replicates=200)
    second = lineage_bootstrap_interval(values, seed=17, replicates=200)
    assert first == second
    assert first[0] <= first[1]


def test_paired_transfer_comparison_uses_identical_episodes() -> None:
    treatment = {
        "e1": (1, 0.8, "l1"),
        "e2": (0, 0.2, "l1"),
        "e3": (1, 0.7, "l2"),
    }
    control = {
        "e1": (1, 0.6, "l1"),
        "e2": (0, 0.4, "l1"),
        "e3": (1, 0.55, "l2"),
    }
    report = paired_transfer_comparison(
        treatment,
        control,
        name="test",
        seed=3,
        replicates=200,
    )
    assert report["records"] == 3
    assert report["lineages"] == 2
    assert report["mean_log_loss_difference"] < 0.0
    assert report["treatment_minus_control_accuracy"] == 0.0


def test_paired_transfer_rejects_metadata_mismatch() -> None:
    with pytest.raises(ValueError, match="metadata differs"):
        paired_transfer_comparison(
            {"e1": (1, 0.8, "l1")},
            {"e1": (0, 0.2, "l1")},
            name="bad",
            seed=1,
            replicates=10,
        )
