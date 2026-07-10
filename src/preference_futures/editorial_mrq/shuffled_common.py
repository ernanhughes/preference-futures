"""Shared contract and deterministic-label helpers for Step 8.7."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.training.common import canonical_json_sha256, load_json, sha256_file

SHUFFLED_CONTROL_SCHEMA_VERSION = 1
SHUFFLE_SEEDS = (1701, 2711, 3719, 4721, 5737)
SHUFFLED_ARMS = ("shuffled_mrq_blind", "shuffled_mrq_choice_aware")
AUTHENTIC_TO_SHUFFLED = {
    "mrq_blind": "shuffled_mrq_blind",
    "mrq_choice_aware": "shuffled_mrq_choice_aware",
}
REQUIRED_NEGATIVE_REPLICATES = 4
BOOTSTRAP_SEED = 17017
BOOTSTRAP_REPLICATES = 10_000


def shuffled_labels_by_partition(
    authentic_labels: Sequence[int],
    partitions: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> list[int]:
    """Permute labels independently within each partition while preserving counts exactly."""

    labels = [int(value) for value in authentic_labels]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("Step 8.7 shuffle requires binary labels")
    expected_indices = set(range(len(labels)))
    observed_indices: set[int] = set()
    output = list(labels)
    for offset, partition in enumerate(("train", "validation", "test")):
        indices = [int(index) for index in partitions.get(partition, ())]
        if not indices:
            raise ValueError(f"Step 8.7 partition is empty: {partition}")
        if observed_indices.intersection(indices):
            raise ValueError("Step 8.7 partitions overlap")
        observed_indices.update(indices)
        values = [labels[index] for index in indices]
        rng = random.Random(seed + (offset + 1) * 100_003)
        rng.shuffle(values)
        for index, value in zip(indices, values, strict=True):
            output[index] = value
        if sum(output[index] for index in indices) != sum(labels[index] for index in indices):
            raise ValueError(f"Step 8.7 shuffle changed class count: {partition}")
    if observed_indices != expected_indices:
        raise ValueError("Step 8.7 partitions do not cover every row exactly once")
    return output


def changed_fraction(
    authentic_labels: Sequence[int],
    shuffled_labels: Sequence[int],
    indices: Sequence[int],
) -> float:
    if not indices:
        raise ValueError("Step 8.7 changed fraction requires rows")
    changed = sum(
        int(authentic_labels[int(index)]) != int(shuffled_labels[int(index)])
        for index in indices
    )
    return changed / len(indices)


def comparison_passed(comparison: Mapping[str, Any]) -> bool:
    interval = comparison["confidence_interval_95"]
    return (
        float(comparison["mean_log_loss_difference"]) < 0.0
        and float(interval[1]) < 0.0
    )


def load_contract(root: Path) -> dict[str, Any]:
    path = root.expanduser().resolve() / "contract.json"
    contract = load_json(path)
    expected = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError("Step 8.7 contract hash is invalid")
    if contract.get("status") != "frozen_before_shuffled_source_training":
        raise ValueError("Step 8.7 contract is not frozen")
    for source in contract.get("sources", {}).values():
        if isinstance(source, Mapping) and source.get("path") and source.get("sha256"):
            path_value = Path(str(source["path"]))
            if not path_value.exists() or sha256_file(path_value) != str(source["sha256"]):
                raise ValueError(f"Step 8.7 source changed: {path_value}")
    return contract


def load_canonical_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    expected = str(report.get("report_sha256", ""))
    payload = dict(report)
    payload.pop("report_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError(f"Step 8.7 report hash is invalid: {path}")
    if report.get("status") != "complete":
        raise ValueError(f"Step 8.7 report is incomplete: {path}")
    return report
