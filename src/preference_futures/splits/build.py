"""Build deterministic train/validation/test manifests grouped by article lineage."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LineageStats:
    lineage_id: str
    episodes: int
    future_revised: int
    selected_b: int
    number_changed: int
    number_dominant: int
    casualty_count: int

    @property
    def future_revised_rate(self) -> float:
        return self.future_revised / self.episodes


@dataclass(slots=True)
class FoldAccumulator:
    lineages: int = 0
    episodes: int = 0
    future_revised: int = 0
    selected_b: int = 0
    number_changed: int = 0
    number_dominant: int = 0
    casualty_count: int = 0

    def add(self, lineage: LineageStats) -> None:
        self.lineages += 1
        self.episodes += lineage.episodes
        self.future_revised += lineage.future_revised
        self.selected_b += lineage.selected_b
        self.number_changed += lineage.number_changed
        self.number_dominant += lineage.number_dominant
        self.casualty_count += lineage.casualty_count


def load_numeric_flags(path: Path) -> dict[str, dict[str, Any]]:
    """Load numeric shortcut flags keyed by episode ID."""

    flags: dict[str, dict[str, Any]] = {}
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid numeric flag JSON on line {line_number}: {exc}") from exc
            episode_id = str(record.get("episode_id", "")).strip()
            if not episode_id:
                raise ValueError(f"numeric flag line {line_number} has no episode_id")
            if episode_id in flags:
                raise ValueError(f"duplicate numeric flag episode_id: {episode_id}")
            flags[episode_id] = record
    return flags


def build_grouped_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    numeric_flags: Mapping[str, Mapping[str, Any]] | None = None,
    folds: int = 10,
    seed: int = 17,
    episodes_path: Path | None = None,
    numeric_flags_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build outer folds with test fold ``i`` and validation fold ``i+1``.

    Each article lineage is assigned to exactly one outer bucket. For fold ``i``:

    - test lineages are bucket ``i``;
    - validation lineages are bucket ``(i + 1) % folds``;
    - every remaining lineage is training data.

    With ten folds this creates an 80/10/10 train/validation/test structure while
    ensuring every lineage is test exactly once and validation exactly once.
    """

    if folds < 3:
        raise ValueError("folds must be at least 3")
    if not records:
        raise ValueError("records must not be empty")

    numeric_flags = numeric_flags or {}
    _validate_episode_records(records)
    _validate_numeric_flags(records, numeric_flags)

    lineage_stats = _build_lineage_stats(records, numeric_flags)
    assignments = _assign_lineages(lineage_stats, folds=folds, seed=seed)
    fold_documents = _build_fold_documents(
        lineage_stats,
        assignments,
        folds=folds,
        seed=seed,
    )

    totals = _summarise_lineages(tuple(lineage_stats.values()))
    outer_summaries = [
        {
            "fold": fold_document["fold"],
            "path": f"fold-{fold_document['fold']:02d}.json",
            "partitions": fold_document["partitions"],
        }
        for fold_document in fold_documents
    ]
    gates = _build_gates(totals, fold_documents, folds=folds)

    manifest = {
        "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "outer_folds": folds,
        "grouping_key": "lineage_id",
        "policy": {
            "test_bucket": "fold i",
            "validation_bucket": "fold (i + 1) mod K",
            "training_buckets": "all remaining folds",
            "expected_partition_shares": {
                "train": (folds - 2) / folds,
                "validation": 1 / folds,
                "test": 1 / folds,
            },
        },
        "sources": {
            "episodes": _source_metadata(episodes_path),
            "numeric_flags": _source_metadata(numeric_flags_path),
        },
        "totals": totals,
        "lineage_to_outer_fold": dict(sorted(assignments.items())),
        "folds": outer_summaries,
        "gates": gates,
        "warnings": _build_warnings(totals, fold_documents),
    }
    return manifest, fold_documents


def write_grouped_split_artifacts(
    output_directory: Path,
    manifest: Mapping[str, Any],
    fold_documents: Sequence[Mapping[str, Any]],
) -> None:
    """Write one manifest, one file per fold, and human-readable summaries."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for fold_document in fold_documents:
        fold = int(fold_document["fold"])
        (output / f"fold-{fold:02d}.json").write_text(
            json.dumps(fold_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        "split_manifest_schema_version": manifest["split_manifest_schema_version"],
        "seed": manifest["seed"],
        "outer_folds": manifest["outer_folds"],
        "totals": manifest["totals"],
        "folds": manifest["folds"],
        "gates": manifest["gates"],
        "warnings": manifest["warnings"],
    }
    (output / "split-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "split-summary.md").write_text(
        render_split_summary_markdown(manifest),
        encoding="utf-8",
    )


def render_split_summary_markdown(manifest: Mapping[str, Any]) -> str:
    """Render fold balance and leakage gates for review and publication."""

    totals = manifest["totals"]
    lines = [
        "# Grouped Split Manifest",
        "",
        "## Dataset",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Episodes | {totals['episodes']:,} |",
        f"| Article lineages | {totals['lineages']:,} |",
        f"| Future revised | {totals['future_revised']:,} |",
        f"| Future-revision rate | {totals['future_revised_rate']:.4f} |",
        f"| Selected-B rate | {totals['selected_b_rate']:.4f} |",
        f"| Number-changed rate | {totals['number_changed_rate']:.4f} |",
        f"| Number-dominant rate | {totals['number_dominant_rate']:.4f} |",
        f"| Casualty-count rate | {totals['casualty_count_rate']:.4f} |",
        "",
        "## Outer test-fold balance",
        "",
        "| Fold | Test lineages | Test episodes | Episode share | Revised rate | Number-changed rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in manifest["folds"]:
        test = fold["partitions"]["test"]
        lines.append(
            "| {fold} | {lineages:,} | {episodes:,} | {share:.4f} | {revised:.4f} | {numeric:.4f} |".format(
                fold=fold["fold"],
                lineages=test["lineages"],
                episodes=test["episodes"],
                share=test["episode_share"],
                revised=test["future_revised_rate"],
                numeric=test["number_changed_rate"],
            )
        )

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in manifest["gates"].items()
    )
    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Experimental consequence",
            "",
            "All downstream encoder training, future-head training, calibration, model selection,",
            "bootstrapping and ablation analysis must consume these lineage assignments rather than",
            "creating new row-level splits.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_episode_records(records: Sequence[Mapping[str, Any]]) -> None:
    seen_episode_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        episode_id = str(record.get("episode_id", "")).strip()
        lineage_id = str(record.get("lineage_id", "")).strip()
        if not episode_id or not lineage_id:
            raise ValueError(f"episode {index} requires episode_id and lineage_id")
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        seen_episode_ids.add(episode_id)
        if type(record.get("future_revised")) is not bool:
            raise TypeError(f"episode {episode_id} future_revised must be bool")
        if record.get("selected_index") not in (0, 1):
            raise ValueError(f"episode {episode_id} selected_index must be 0 or 1")


def _validate_numeric_flags(
    records: Sequence[Mapping[str, Any]],
    numeric_flags: Mapping[str, Mapping[str, Any]],
) -> None:
    if not numeric_flags:
        return
    episode_ids = {str(record["episode_id"]) for record in records}
    missing = sorted(episode_ids.difference(numeric_flags))
    extras = sorted(set(numeric_flags).difference(episode_ids))
    if missing:
        raise ValueError(f"numeric flags missing {len(missing)} episode IDs; first: {missing[0]}")
    if extras:
        raise ValueError(f"numeric flags contain {len(extras)} unknown episode IDs; first: {extras[0]}")


def _build_lineage_stats(
    records: Sequence[Mapping[str, Any]],
    numeric_flags: Mapping[str, Mapping[str, Any]],
) -> dict[str, LineageStats]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["lineage_id"])].append(record)

    result: dict[str, LineageStats] = {}
    for lineage_id, lineage_records in grouped.items():
        result[lineage_id] = LineageStats(
            lineage_id=lineage_id,
            episodes=len(lineage_records),
            future_revised=sum(bool(record["future_revised"]) for record in lineage_records),
            selected_b=sum(int(record["selected_index"]) == 1 for record in lineage_records),
            number_changed=sum(
                bool(numeric_flags.get(str(record["episode_id"]), {}).get("number_changed"))
                for record in lineage_records
            ),
            number_dominant=sum(
                bool(
                    numeric_flags.get(str(record["episode_id"]), {}).get(
                        "number_dominant_edit"
                    )
                )
                for record in lineage_records
            ),
            casualty_count=sum(
                bool(
                    numeric_flags.get(str(record["episode_id"]), {}).get(
                        "casualty_count_update"
                    )
                )
                for record in lineage_records
            ),
        )
    return result


def _assign_lineages(
    lineage_stats: Mapping[str, LineageStats],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    totals = _summarise_lineages(tuple(lineage_stats.values()))
    targets = {
        "lineages": totals["lineages"] / folds,
        "episodes": totals["episodes"] / folds,
        "future_revised": totals["future_revised"] / folds,
        "selected_b": totals["selected_b"] / folds,
        "number_changed": totals["number_changed"] / folds,
        "number_dominant": totals["number_dominant"] / folds,
        "casualty_count": totals["casualty_count"] / folds,
    }
    global_rate = totals["future_revised_rate"]
    ordered = sorted(
        lineage_stats.values(),
        key=lambda lineage: (
            -lineage.episodes,
            -abs(lineage.future_revised_rate - global_rate) * lineage.episodes,
            _stable_hash(seed, lineage.lineage_id),
        ),
    )

    accumulators = [FoldAccumulator() for _ in range(folds)]
    assignments: dict[str, int] = {}
    for lineage in ordered:
        candidate_folds = sorted(
            range(folds),
            key=lambda fold: _stable_hash(seed, lineage.lineage_id, str(fold)),
        )
        selected_fold = min(
            candidate_folds,
            key=lambda fold: _projected_balance_score(
                accumulators[fold],
                lineage,
                targets,
            ),
        )
        accumulators[selected_fold].add(lineage)
        assignments[lineage.lineage_id] = selected_fold
    return assignments


def _projected_balance_score(
    accumulator: FoldAccumulator,
    lineage: LineageStats,
    targets: Mapping[str, float],
) -> float:
    projected = {
        "lineages": accumulator.lineages + 1,
        "episodes": accumulator.episodes + lineage.episodes,
        "future_revised": accumulator.future_revised + lineage.future_revised,
        "selected_b": accumulator.selected_b + lineage.selected_b,
        "number_changed": accumulator.number_changed + lineage.number_changed,
        "number_dominant": accumulator.number_dominant + lineage.number_dominant,
        "casualty_count": accumulator.casualty_count + lineage.casualty_count,
    }
    weights = {
        "lineages": 0.35,
        "episodes": 1.0,
        "future_revised": 1.0,
        "selected_b": 0.25,
        "number_changed": 0.35,
        "number_dominant": 0.20,
        "casualty_count": 0.10,
    }
    score = 0.0
    for field, weight in weights.items():
        target = max(targets[field], 1.0)
        score += weight * ((projected[field] - target) / target) ** 2
    return score


def _build_fold_documents(
    lineage_stats: Mapping[str, LineageStats],
    assignments: Mapping[str, int],
    *,
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    all_lineages = set(lineage_stats)
    documents = []
    for fold in range(folds):
        test_lineages = {
            lineage_id for lineage_id, assigned_fold in assignments.items() if assigned_fold == fold
        }
        validation_fold = (fold + 1) % folds
        validation_lineages = {
            lineage_id
            for lineage_id, assigned_fold in assignments.items()
            if assigned_fold == validation_fold
        }
        train_lineages = all_lineages.difference(test_lineages, validation_lineages)
        _assert_disjoint_complete(
            all_lineages,
            train_lineages,
            validation_lineages,
            test_lineages,
        )
        documents.append(
            {
                "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
                "fold": fold,
                "seed": seed,
                "test_outer_fold": fold,
                "validation_outer_fold": validation_fold,
                "train_lineages": sorted(train_lineages),
                "validation_lineages": sorted(validation_lineages),
                "test_lineages": sorted(test_lineages),
                "partitions": {
                    "train": _partition_summary(train_lineages, lineage_stats, len(all_lineages)),
                    "validation": _partition_summary(
                        validation_lineages,
                        lineage_stats,
                        len(all_lineages),
                    ),
                    "test": _partition_summary(test_lineages, lineage_stats, len(all_lineages)),
                },
            }
        )
    return documents


def _partition_summary(
    lineages: set[str],
    lineage_stats: Mapping[str, LineageStats],
    total_lineages: int,
) -> dict[str, Any]:
    selected = tuple(lineage_stats[lineage_id] for lineage_id in lineages)
    summary = _summarise_lineages(selected)
    total_episodes = sum(lineage.episodes for lineage in lineage_stats.values())
    summary["lineage_share"] = len(lineages) / total_lineages
    summary["episode_share"] = summary["episodes"] / total_episodes
    return summary


def _summarise_lineages(lineages: Sequence[LineageStats]) -> dict[str, Any]:
    episodes = sum(lineage.episodes for lineage in lineages)
    future_revised = sum(lineage.future_revised for lineage in lineages)
    selected_b = sum(lineage.selected_b for lineage in lineages)
    number_changed = sum(lineage.number_changed for lineage in lineages)
    number_dominant = sum(lineage.number_dominant for lineage in lineages)
    casualty_count = sum(lineage.casualty_count for lineage in lineages)
    return {
        "lineages": len(lineages),
        "episodes": episodes,
        "future_revised": future_revised,
        "future_stable": episodes - future_revised,
        "future_revised_rate": future_revised / episodes if episodes else 0.0,
        "selected_b": selected_b,
        "selected_b_rate": selected_b / episodes if episodes else 0.0,
        "number_changed": number_changed,
        "number_changed_rate": number_changed / episodes if episodes else 0.0,
        "number_dominant": number_dominant,
        "number_dominant_rate": number_dominant / episodes if episodes else 0.0,
        "casualty_count": casualty_count,
        "casualty_count_rate": casualty_count / episodes if episodes else 0.0,
    }


def _assert_disjoint_complete(
    all_lineages: set[str],
    train: set[str],
    validation: set[str],
    test: set[str],
) -> None:
    if train.intersection(validation) or train.intersection(test) or validation.intersection(test):
        raise AssertionError("lineage leakage detected between partitions")
    if train.union(validation, test) != all_lineages:
        raise AssertionError("partition union does not cover every lineage")


def _build_gates(
    totals: Mapping[str, Any],
    fold_documents: Sequence[Mapping[str, Any]],
    *,
    folds: int,
) -> dict[str, bool]:
    expected_share = 1 / folds
    test_partitions = [fold["partitions"]["test"] for fold in fold_documents]
    validation_partitions = [fold["partitions"]["validation"] for fold in fold_documents]
    assignments_tested_once = sum(partition["lineages"] for partition in test_partitions) == totals[
        "lineages"
    ]
    assignments_validated_once = sum(
        partition["lineages"] for partition in validation_partitions
    ) == totals["lineages"]
    max_episode_share_deviation = max(
        abs(partition["episode_share"] - expected_share) for partition in test_partitions
    )
    max_lineage_share_deviation = max(
        abs(partition["lineage_share"] - expected_share) for partition in test_partitions
    )
    max_target_rate_deviation = max(
        abs(partition["future_revised_rate"] - totals["future_revised_rate"])
        for partition in test_partitions
    )
    max_numeric_rate_deviation = max(
        abs(partition["number_changed_rate"] - totals["number_changed_rate"])
        for partition in test_partitions
    )
    return {
        "all_lineages_tested_exactly_once": assignments_tested_once,
        "all_lineages_validated_exactly_once": assignments_validated_once,
        "test_episode_share_within_2_points": max_episode_share_deviation <= 0.02,
        "test_lineage_share_within_2_points": max_lineage_share_deviation <= 0.02,
        "test_future_rate_within_3_points": max_target_rate_deviation <= 0.03,
        "test_numeric_change_rate_within_3_points": max_numeric_rate_deviation <= 0.03,
    }


def _build_warnings(
    totals: Mapping[str, Any],
    fold_documents: Sequence[Mapping[str, Any]],
) -> list[str]:
    warnings = []
    test_partitions = [fold["partitions"]["test"] for fold in fold_documents]
    max_target_rate_deviation = max(
        abs(partition["future_revised_rate"] - totals["future_revised_rate"])
        for partition in test_partitions
    )
    if max_target_rate_deviation >= 0.015:
        warnings.append(
            "At least one outer test fold differs from the global future-revision rate by 1.5 "
            "percentage points or more; retain paired fold-level reporting."
        )
    if totals["number_changed"] == 0:
        warnings.append(
            "No numeric flags were supplied; numeric shortcut balance was not actively stratified."
        )
    warnings.append(
        "Do not tune split assignments after observing model outcomes; regenerate only for a "
        "preregistered sensitivity analysis with a different seed."
    )
    return warnings


def _stable_hash(seed: int, *values: str) -> int:
    payload = "\x1f".join((str(seed), *values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _source_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }
