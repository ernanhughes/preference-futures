"""Deterministic binary forecast metrics used by Step 6."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from preference_futures.probes.common import PROBABILITY_EPSILON


def binary_metrics(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    clean_labels = [int(label) for label in labels]
    if any(label not in (0, 1) for label in clean_labels):
        raise ValueError("binary labels must be zero or one")
    clean_probabilities = [float(probability) for probability in probabilities]
    if any(not math.isfinite(probability) for probability in clean_probabilities):
        raise ValueError("probabilities must be finite")
    if any(probability < 0.0 or probability > 1.0 for probability in clean_probabilities):
        raise ValueError("probabilities must be in [0, 1]")

    clipped = [
        min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, probability))
        for probability in clean_probabilities
    ]
    losses = [
        -(label * math.log(probability) + (1 - label) * math.log(1.0 - probability))
        for label, probability in zip(clean_labels, clipped, strict=True)
    ]
    squared_errors = [
        (probability - label) ** 2
        for label, probability in zip(clean_labels, clean_probabilities, strict=True)
    ]
    predictions = [int(probability >= 0.5) for probability in clean_probabilities]
    positives = sum(clean_labels)
    total = len(clean_labels)
    return {
        "records": total,
        "positives": positives,
        "prevalence": positives / total,
        "mean_probability": sum(clean_probabilities) / total,
        "log_loss": sum(losses) / total,
        "brier_score": sum(squared_errors) / total,
        "accuracy": sum(
            prediction == label
            for prediction, label in zip(predictions, clean_labels, strict=True)
        )
        / total,
        "roc_auc": roc_auc(clean_labels, clean_probabilities),
    }


def per_record_log_losses(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
) -> list[float]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    result = []
    for label, probability in zip(labels, probabilities, strict=True):
        clipped = min(
            1.0 - PROBABILITY_EPSILON,
            max(PROBABILITY_EPSILON, float(probability)),
        )
        target = int(label)
        result.append(
            -(target * math.log(clipped) + (1 - target) * math.log(1.0 - clipped))
        )
    return result


def roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(
        enumerate(probabilities),
        key=lambda item: (float(item[1]), item[0]),
    )
    ranks = [0.0] * len(labels)
    index = 0
    while index < len(ordered):
        end = index + 1
        value = float(ordered[index][1])
        while end < len(ordered) and float(ordered[end][1]) == value:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
