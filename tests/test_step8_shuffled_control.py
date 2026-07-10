from __future__ import annotations

import pytest

from preference_futures.editorial_mrq.shuffled_aggregate import (
    compare_authentic_to_mean_shuffled,
)
from preference_futures.editorial_mrq.shuffled_common import (
    changed_fraction,
    comparison_passed,
    shuffled_labels_by_partition,
)


def test_partition_shuffle_is_deterministic_and_preserves_counts() -> None:
    labels = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    partitions = {
        "train": [0, 1, 2, 3, 4, 5],
        "validation": [6, 7, 8],
        "test": [9, 10, 11],
    }
    first = shuffled_labels_by_partition(labels, partitions, seed=17)
    second = shuffled_labels_by_partition(labels, partitions, seed=17)
    assert first == second
    for indices in partitions.values():
        assert sum(first[index] for index in indices) == sum(labels[index] for index in indices)
    assert 0.0 <= changed_fraction(labels, first, partitions["train"]) <= 1.0


def test_partition_shuffle_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        shuffled_labels_by_partition(
            [0, 1, 0],
            {"train": [0, 1], "validation": [1], "test": [2]},
            seed=3,
        )


def test_authentic_to_mean_shuffled_comparison() -> None:
    authentic = {
        "e1": (1, 0.9, "l1"),
        "e2": (0, 0.1, "l1"),
        "e3": (1, 0.8, "l2"),
        "e4": (0, 0.2, "l3"),
    }
    controls = [
        {
            "e1": (1, 0.6, "l1"),
            "e2": (0, 0.4, "l1"),
            "e3": (1, 0.55, "l2"),
            "e4": (0, 0.45, "l3"),
        },
        {
            "e1": (1, 0.65, "l1"),
            "e2": (0, 0.35, "l1"),
            "e3": (1, 0.6, "l2"),
            "e4": (0, 0.4, "l3"),
        },
    ]
    report = compare_authentic_to_mean_shuffled(
        authentic,
        controls,
        name="test",
        seed=5,
        replicates=200,
    )
    assert report["records"] == 4
    assert report["shuffled_replicates"] == 2
    assert report["mean_log_loss_difference"] < 0.0
    assert report["authentic_log_loss"] < report["mean_shuffled_log_loss"]


def test_comparison_gate_requires_interval_below_zero() -> None:
    assert comparison_passed(
        {"mean_log_loss_difference": -0.01, "confidence_interval_95": [-0.02, -0.001]}
    )
    assert not comparison_passed(
        {"mean_log_loss_difference": -0.01, "confidence_interval_95": [-0.02, 0.001]}
    )
