"""Build deterministic train/validation/test manifests grouped by article lineage."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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

    def projected(self, lineage: LineageStats | None = None) -> dict[str, int]:
        return {
            "lineages": self.lineages + (1 if lineage else 0),
            "episodes": self.episodes + (lineage.episodes if lineage else 0),
            "future_revised": self.future_revised + (lineage.future_revised if lineage else 0),
            "selected_b": self.selected_b + (lineage.selected_b if lineage else 0),
            "number_changed": self.number_changed + (lineage.number_changed if lineage else 0),
            "number_dominant": self.number_dominant + (lineage.number_dominant if lineage else 0),
            "casualty_count": self.casualty_count + (lineage.casualty_count if lineage else 0),
        }


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
    """Build deterministic lineage-grouped outer folds.

    Test uses bucket ``i``; validation uses bucket ``(i + 1) % folds``; every
    remaining bucket is training data. Ten folds therefore produce 80/10/10
    partitions while testing and validating every lineage exactly once.
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
    diagnostics = _balance_diagnostics(totals, fold_documents, folds=folds)
    gates = _build_gates(diagnostics)

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
        "folds": [
            {
                "fold": document["fold"],
                "path": f"fold-{int(document['fold']):02d}.json",
                "partitions": document["partitions"],
            }
            for document in fold_documents
        ],
        "balance_diagnostics": diagnostics,
        "gates": gates,
        "warnings": _build_warnings(totals, diagnostics),
    }
    return manifest, fold_documents


def write_grouped_split_artifacts(
    output_directory: Path,
    manifest: Mapping[str, Any],
    fold_documents: Sequence[Mapping[str, Any]],
) -> None:
    """Write one manifest, one file per fold, and compact review summaries."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for document in fold_documents:
        fold = int(document["fold"])
        (output / f"fold-{fold:02d}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        key: manifest[key]
        for key in (
            "split_manifest_schema_version",
            "seed",
            "outer_folds",
            "sources",
            "totals",
            "folds",
            "balance_diagnostics",
            "gates",
            "warnings",
        )
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
    diagnostics = manifest["balance_diagnostics"]
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
        "## Maximum outer test-fold deviations",
        "",
        "| Measure | Absolute deviation |",
        "|---|---:|",
        f"| Episode share from expected | {diagnostics['max_test_episode_share_deviation']:.6f} |",
        f"| Lineage share from expected | {diagnostics['max_test_lineage_share_deviation']:.6f} |",
        f"| Future-revision rate from global | {diagnostics['max_test_future_rate_deviation']:.6f} |",
        f"| Numerical-change rate from global | {diagnostics['max_test_numeric_rate_deviation']:.6f} |",
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

    lines.extend(["", "## Gates", "", "| Gate | Result |", "|---|---|"])
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in manifest["gates"].items()
    )
    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Experimental consequence",
            "",
            "All encoder training, future-head training, calibration, model selection,",
            "bootstrapping and ablation analysis must consume these lineage assignments.",
            "No downstream stage may create a new row-level split.",
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
        if type(record.get("selected_index")) is not int or record["selected_index"] not in (0, 1):
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

    stats: dict[str, LineageStats] = {}
    for lineage_id, lineage_records in grouped.items():
        stats[lineage_id] = LineageStats(
            lineage_id=lineage_id,
            episodes=len(lineage_records),
            future_revised=sum(bool(record["future_revised"]) for record in lineage_records),
            selected_b=sum(int(record["selected_index"]) == 1 for record in lineage_records),
            number_changed=_flag_count(lineage_records, numeric_flags, "number_changed"),
            number_dominant=_flag_count(
                lineage_records,
                numeric_flags,
                "number_dominant_edit",
            ),
            casualty_count=_flag_count(
                lineage_records,
                numeric_flags,
                "casualty_count_update",
            ),
        )
    return stats


def _flag_count(
    records: Sequence[Mapping[str, Any]],
    numeric_flags: Mapping[str, Mapping[str, Any]],
    flag: str,
) -> int:
    return sum(
        bool(numeric_flags.get(str(record["episode_id"]), {}).get(flag)) for record in records
    )


def _assign_lineages(
    lineage_stats: Mapping[str, LineageStats],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    totals = _summarise_lineages(tuple(lineage_stats.values()))
    targets = {
        field: totals[field] / folds
        for field in (
            "lineages",
            "episodes",
            "future_revised",
            "selected_b",
            "number_changed",
            "number_dominant",
            "casualty_count",
        )
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
            key=lambda fold: _global_projected_score(
                accumulators,
                candidate_fold=fold,
                lineage=lineage,
                targets=targets,
            ),
        )
        accumulators[selected_fold].add(lineage)
        assignments[lineage.lineage_id] = selected_fold
    return assignments


def _global_projected_score(
    accumulators: Sequence[FoldAccumulator],
    *,
    candidate_fold: int,
    lineage: LineageStats,
    targets: Mapping[str, float],
) -> float:
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
    for fold, accumulator in enumerate(accumulators):
        values = accumulator.projected(lineage if fold == candidate_fold else None)
        for field, weight in weights.items():
            target = max(targets[field], 1.0)
            score += weight * ((values[field] - target) / target) ** 2
    return score


def _build_fold_documents(
    lineage_stats: Mapping[str, LineageStats],
    assignments: Mapping[str, int],
    *,
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    all_lineages = set(lineage_stats)
    documents: list[dict[str, Any]] = []
    for fold in range(folds):
        test = {lineage for lineage, bucket in assignments.items() if bucket == fold}
        validation_bucket = (fold + 1) % folds
        validation = {
            lineage for lineage, bucket in assignments.items() if bucket == validation_bucket
        }
        train = all_lineages.difference(test, validation)
        _assert_disjoint_complete(all_lineages, train, validation, test)
        documents.append(
            {
                "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
                "fold": fold,
                "seed": seed,
                "test_outer_fold": fold,
                "validation_outer_fold": validation_bucket,
                "train_lineages": sorted(train),
                "validation_lineages": sorted(validation),
                "test_lineages": sorted(test),
                "partitions": {
                    "train": _partition_summary(train, lineage_stats),
                    "validation": _partition_summary(validation, lineage_stats),
                    "test": _partition_summary(test, lineage_stats),
                },
            }
        )
    return documents


def _partition_summary(
    lineages: set[str],
    lineage_stats: Mapping[str, LineageStats],
) -> dict[str, Any]:
    selected = tuple(lineage_stats[lineage] for lineage in lineages)
    summary = _summarise_lineages(selected)
    all_stats = tuple(lineage_stats.values())
    total_episodes = sum(lineage.episodes for lineage in all_stats)
    summary["lineage_share"] = len(lineages) / len(lineage_stats)
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


def _balance_diagnostics(
    totals: Mapping[str, Any],
    fold_documents: Sequence[Mapping[str, Any]],
    *,
    folds: int,
) -> dict[str, Any]:
    expected_share = 1 / folds
    tests = [document["partitions"]["test"] for document in fold_documents]
    validations = [document["partitions"]["validation"] for document in fold_documents]
    return {
        "test_lineage_assignments": sum(partition["lineages"] for partition in tests),
        "validation_lineage_assignments": sum(
            partition["lineages"] for partition in validations
        ),
        "max_test_episode_share_deviation": max(
            abs(partition["episode_share"] - expected_share) for partition in tests
        ),
        "max_test_lineage_share_deviation": max(
            abs(partition["lineage_share"] - expected_share) for partition in tests
        ),
        "max_test_future_rate_deviation": max(
            abs(partition["future_revised_rate"] - totals["future_revised_rate"])
            for partition in tests
        ),
        "max_test_numeric_rate_deviation": max(
            abs(partition["number_changed_rate"] - totals["number_changed_rate"])
            for partition in tests
        ),
    }


def _build_gates(diagnostics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "all_lineages_tested_exactly_once": (
            diagnostics["test_lineage_assignments"]
            == diagnostics["validation_lineage_assignments"]
        ),
        "all_lineages_validated_exactly_once": (
            diagnostics["validation_lineage_assignments"]
            == diagnostics["test_lineage_assignments"]
        ),
        "test_episode_share_within_2_points": (
            diagnostics["max_test_episode_share_deviation"] <= 0.02
        ),
        "test_lineage_share_within_2_points": (
            diagnostics["max_test_lineage_share_deviation"] <= 0.02
        ),
        "test_future_rate_within_3_points": (
            diagnostics["max_test_future_rate_deviation"] <= 0.03
        ),
        "test_numeric_change_rate_within_3_points": (
            diagnostics["max_test_numeric_rate_deviation"] <= 0.03
        ),
    }


def _build_warnings(
    totals: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if diagnostics["max_test_future_rate_deviation"] >= 0.015:
        warnings.append(
            "At least one outer test fold differs from the global future-revision rate by 1.5 "
            "percentage points or more; retain paired fold-level reporting."
        )
    if totals["number_changed"] == 0:
        warnings.append(
            "No numeric flags were supplied; numeric shortcut balance was not actively stratified."
        )
    warnings.append(
        "Do not tune assignments after observing model outcomes; other seeds are sensitivity "
        "analyses, not opportunities to select a favourable result."
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
