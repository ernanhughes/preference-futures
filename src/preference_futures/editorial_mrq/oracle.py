"""Candidate-order invariance utilities for the Step 8 oracle audit."""

from __future__ import annotations

import math
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.training.common import (
    canonical_json_sha256,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

SWAPPED_ORACLE_SCHEMA_VERSION = 1


def export_swapped_oracle_prompts(
    *,
    prompts_path: Path,
    output_directory: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Swap candidate A/B without opening an answer key."""

    source = prompts_path.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    _prepare_output(output, force=force)
    prompts = load_jsonl(source)
    swapped: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in prompts:
        item_id = int(record["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate oracle item_id: {item_id}")
        seen.add(item_id)
        candidate_a = str(record["candidate_a"])
        candidate_b = str(record["candidate_b"])
        swapped.append(
            {
                **dict(record),
                "oracle_swap_schema_version": SWAPPED_ORACLE_SCHEMA_VERSION,
                "item_id": item_id,
                "candidate_a": candidate_b,
                "candidate_b": candidate_a,
                "candidate_order_swapped": True,
            }
        )
    destination = output / "oracle-prompts-swapped.jsonl"
    write_jsonl(destination, swapped)
    manifest: dict[str, Any] = {
        "oracle_swap_schema_version": SWAPPED_ORACLE_SCHEMA_VERSION,
        "status": "complete",
        "items": len(swapped),
        "source_prompts_path": str(source),
        "source_prompts_sha256": sha256_file(source),
        "swapped_prompts_path": str(destination),
        "swapped_prompts_sha256": sha256_file(destination),
        "answer_key_opened": False,
        "transformation": "candidate_a_and_candidate_b_exchanged",
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_json(output / "oracle-swap-manifest.json", manifest)
    return manifest


def score_swapped_oracle_predictions(
    *,
    original_predictions_path: Path,
    swapped_predictions_path: Path,
    answer_key_path: Path,
    output_directory: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Score original and swapped predictions after translating swapped labels back."""

    output = output_directory.expanduser().resolve()
    _prepare_output(output, force=force)
    original = _load_predictions(original_predictions_path)
    swapped = _load_predictions(swapped_predictions_path)
    answers = _load_answers(answer_key_path)
    identifiers = sorted(answers)
    if set(original) != set(identifiers) or set(swapped) != set(identifiers):
        raise ValueError("original, swapped and answer-key item IDs must match exactly")

    rows: list[dict[str, Any]] = []
    original_correct = 0
    swapped_correct = 0
    consistent = 0
    consensus_correct = 0
    consensus_items = 0
    original_counts: Counter[str] = Counter()
    swapped_counts: Counter[str] = Counter()
    for item_id in identifiers:
        original_prediction = original[item_id]
        raw_swapped_prediction = swapped[item_id]
        translated_swapped = _opposite(raw_swapped_prediction)
        answer = answers[item_id]
        is_consistent = original_prediction == translated_swapped
        original_is_correct = original_prediction == answer
        swapped_is_correct = translated_swapped == answer
        original_correct += int(original_is_correct)
        swapped_correct += int(swapped_is_correct)
        consistent += int(is_consistent)
        original_counts[original_prediction] += 1
        swapped_counts[raw_swapped_prediction] += 1
        if is_consistent:
            consensus_items += 1
            consensus_correct += int(original_is_correct)
        rows.append(
            {
                "item_id": item_id,
                "answer": answer,
                "original_prediction": original_prediction,
                "swapped_prediction_raw": raw_swapped_prediction,
                "swapped_prediction_translated": translated_swapped,
                "order_consistent": is_consistent,
                "original_correct": original_is_correct,
                "swapped_correct": swapped_is_correct,
            }
        )

    total = len(identifiers)
    report: dict[str, Any] = {
        "oracle_swap_schema_version": SWAPPED_ORACLE_SCHEMA_VERSION,
        "status": "complete",
        "items": total,
        "original": _accuracy_record(original_correct, total),
        "swapped_translated": _accuracy_record(swapped_correct, total),
        "order_consistency": {
            "consistent": consistent,
            "rate": consistent / max(1, total),
        },
        "consistent_subset": _accuracy_record(consensus_correct, consensus_items),
        "prediction_balance": {
            "original": {"A": original_counts["A"], "B": original_counts["B"]},
            "swapped_raw": {"A": swapped_counts["A"], "B": swapped_counts["B"]},
        },
        "artifacts": {
            "original_predictions_sha256": sha256_file(
                original_predictions_path.expanduser().resolve()
            ),
            "swapped_predictions_sha256": sha256_file(
                swapped_predictions_path.expanduser().resolve()
            ),
            "answer_key_sha256": sha256_file(answer_key_path.expanduser().resolve()),
        },
        "interpretation": {
            "primary_metric": "accuracy_on_order_consistent_subset",
            "swapped_predictions_translated_back_to_original_orientation": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output / "oracle-swap-score.json", report)
    write_jsonl(output / "oracle-swap-rows.jsonl", rows)
    (output / "oracle-swap-score.md").write_text(
        render_swapped_oracle_markdown(report),
        encoding="utf-8",
    )
    return report


def render_swapped_oracle_markdown(report: Mapping[str, Any]) -> str:
    original = report["original"]
    swapped = report["swapped_translated"]
    consistent = report["order_consistency"]
    subset = report["consistent_subset"]
    return "\n".join(
        [
            "# Step 8.0 Candidate-Swap Oracle Audit",
            "",
            f"- Items: `{int(report['items'])}`",
            f"- Original accuracy: `{float(original['accuracy']):.6f}`",
            f"- Swapped translated accuracy: `{float(swapped['accuracy']):.6f}`",
            f"- Order consistency: `{float(consistent['rate']):.6f}`",
            f"- Consistent-subset items: `{int(subset['items'])}`",
            f"- Consistent-subset accuracy: `{float(subset['accuracy']):.6f}`",
            "",
            "The consistent-subset score is the primary bias-corrected oracle diagnostic.",
            "",
        ]
    )


def _load_predictions(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    for record in load_jsonl(path.expanduser().resolve()):
        item_id = int(record["item_id"])
        prediction = str(record["prediction"]).strip().upper()
        if prediction not in {"A", "B"} or item_id in values:
            raise ValueError(f"invalid or duplicate prediction for item {item_id}")
        values[item_id] = prediction
    if not values:
        raise ValueError("prediction file is empty")
    return values


def _load_answers(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    for record in load_jsonl(path.expanduser().resolve()):
        item_id = int(record["item_id"])
        selected = str(record.get("selected", "")).strip().upper()
        if not selected:
            selected_index = int(record["selected_index"])
            selected = "A" if selected_index == 0 else "B"
        if selected not in {"A", "B"} or item_id in values:
            raise ValueError(f"invalid or duplicate answer for item {item_id}")
        values[item_id] = selected
    if not values:
        raise ValueError("answer-key file is empty")
    return values


def _accuracy_record(correct: int, total: int) -> dict[str, Any]:
    if total < 1:
        return {
            "items": 0,
            "correct": 0,
            "accuracy": 0.0,
            "wilson_interval_95": [0.0, 1.0],
        }
    lower, upper = _wilson_interval(correct, total)
    return {
        "items": total,
        "correct": correct,
        "accuracy": correct / total,
        "wilson_interval_95": [lower, upper],
    }


def _opposite(value: str) -> str:
    return "B" if value == "A" else "A"


def _wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (rate + z2 / (2.0 * total)) / denominator
    spread = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _prepare_output(output: Path, *, force: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"output directory is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
