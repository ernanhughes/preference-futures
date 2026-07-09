"""Independent verification for persisted grouped split manifests."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COUNT_FIELDS = (
    "lineages",
    "episodes",
    "future_revised",
    "future_stable",
    "selected_b",
    "number_changed",
    "number_dominant",
    "casualty_count",
)


def verify_grouped_split_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify persisted assignments independently of the split builder's gates."""

    errors: list[str] = []
    checks: dict[str, bool] = {}

    totals = _mapping(manifest.get("totals"), "totals", errors)
    assignments = _mapping(
        manifest.get("lineage_to_outer_fold"),
        "lineage_to_outer_fold",
        errors,
    )
    fold_documents = _sequence(manifest.get("folds"), "folds", errors)
    outer_folds = manifest.get("outer_folds")

    valid_outer_fold_count = isinstance(outer_folds, int) and not isinstance(outer_folds, bool)
    valid_outer_fold_count = valid_outer_fold_count and outer_folds >= 3
    checks["valid_outer_fold_count"] = valid_outer_fold_count
    if not valid_outer_fold_count:
        errors.append("outer_folds must be an integer of at least 3")
        outer_folds = 0

    total_lineages = totals.get("lineages")
    checks["assignment_count_matches_total_lineages"] = (
        isinstance(total_lineages, int)
        and not isinstance(total_lineages, bool)
        and len(assignments) == total_lineages
    )
    if not checks["assignment_count_matches_total_lineages"]:
        errors.append(
            "lineage_to_outer_fold size does not match totals.lineages: "
            f"{len(assignments)} != {total_lineages}"
        )

    valid_assignment_values = bool(assignments) and outer_folds > 0
    if valid_assignment_values:
        valid_assignment_values = all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < outer_folds
            for value in assignments.values()
        )
    checks["all_assignment_fold_ids_valid"] = valid_assignment_values
    if not valid_assignment_values:
        errors.append("one or more lineage assignments use an invalid outer fold ID")

    folds_by_id: dict[int, Mapping[str, Any]] = {}
    for index, document in enumerate(fold_documents):
        if not isinstance(document, Mapping):
            errors.append(f"folds[{index}] is not an object")
            continue
        fold_id = document.get("fold")
        if not isinstance(fold_id, int) or isinstance(fold_id, bool):
            errors.append(f"folds[{index}].fold is not an integer")
            continue
        if fold_id in folds_by_id:
            errors.append(f"duplicate fold summary for fold {fold_id}")
            continue
        folds_by_id[fold_id] = document

    expected_fold_ids = set(range(outer_folds))
    checks["fold_summaries_cover_each_outer_fold_once"] = set(folds_by_id) == expected_fold_ids
    if not checks["fold_summaries_cover_each_outer_fold_once"]:
        errors.append(
            "fold summary IDs do not match the expected range: "
            f"found={sorted(folds_by_id)}, expected={sorted(expected_fold_ids)}"
        )

    assignment_counts = Counter(assignments.values())
    mapping_matches_summaries = bool(folds_by_id) and set(folds_by_id) == expected_fold_ids
    test_summaries: dict[int, Mapping[str, Any]] = {}
    validation_summaries: dict[int, Mapping[str, Any]] = {}
    train_summaries: dict[int, Mapping[str, Any]] = {}

    for fold_id, document in folds_by_id.items():
        partitions = _mapping(document.get("partitions"), f"fold {fold_id} partitions", errors)
        test = _mapping(partitions.get("test"), f"fold {fold_id} test", errors)
        validation = _mapping(
            partitions.get("validation"),
            f"fold {fold_id} validation",
            errors,
        )
        train = _mapping(partitions.get("train"), f"fold {fold_id} train", errors)
        test_summaries[fold_id] = test
        validation_summaries[fold_id] = validation
        train_summaries[fold_id] = train
        mapping_matches_summaries = mapping_matches_summaries and (
            test.get("lineages") == assignment_counts.get(fold_id, 0)
        )

    checks["assignment_map_matches_test_fold_lineage_counts"] = mapping_matches_summaries
    if not mapping_matches_summaries:
        errors.append("assignment-map fold counts do not match test partition lineage counts")

    for prefix, summaries in (
        ("test", test_summaries),
        ("validation", validation_summaries),
    ):
        for field in _COUNT_FIELDS:
            observed = sum(_integer(summary.get(field)) for summary in summaries.values())
            expected = totals.get(field)
            passed = isinstance(expected, int) and not isinstance(expected, bool) and observed == expected
            check_name = f"{prefix}_{field}_sum_matches_total"
            checks[check_name] = passed
            if not passed:
                errors.append(f"{check_name}: observed {observed}, expected {expected}")

    validation_is_next_test = bool(folds_by_id) and outer_folds > 0
    if validation_is_next_test:
        for fold_id in range(outer_folds):
            validation = validation_summaries.get(fold_id, {})
            next_test = test_summaries.get((fold_id + 1) % outer_folds, {})
            if any(validation.get(field) != next_test.get(field) for field in _COUNT_FIELDS):
                validation_is_next_test = False
                break
    checks["validation_partition_is_next_outer_test_bucket"] = validation_is_next_test
    if not validation_is_next_test:
        errors.append("a validation partition does not equal the next outer test bucket")

    train_is_complement = bool(folds_by_id) and outer_folds > 0
    if train_is_complement:
        for fold_id in range(outer_folds):
            train = train_summaries.get(fold_id, {})
            validation = validation_summaries.get(fold_id, {})
            test = test_summaries.get(fold_id, {})
            for field in _COUNT_FIELDS:
                expected_total = totals.get(field)
                if not isinstance(expected_total, int) or isinstance(expected_total, bool):
                    train_is_complement = False
                    break
                expected_train = expected_total - _integer(validation.get(field)) - _integer(
                    test.get(field)
                )
                if train.get(field) != expected_train:
                    train_is_complement = False
                    break
            if not train_is_complement:
                break
    checks["train_partition_is_exact_complement"] = train_is_complement
    if not train_is_complement:
        errors.append("a training partition is not the exact complement of validation and test")

    source_hashes_valid = True
    sources = manifest.get("sources")
    if isinstance(sources, Mapping):
        for name in ("episodes", "numeric_flags"):
            source = sources.get(name)
            if source is None:
                continue
            if not isinstance(source, Mapping) or not _SHA256_PATTERN.fullmatch(
                str(source.get("sha256", ""))
            ):
                source_hashes_valid = False
                errors.append(f"sources.{name}.sha256 is missing or invalid")
    else:
        source_hashes_valid = False
        errors.append("sources is missing or is not an object")
    checks["source_sha256_values_valid"] = source_hashes_valid

    recorded_gates = manifest.get("gates")
    checks["all_recorded_builder_gates_passed"] = isinstance(recorded_gates, Mapping) and bool(
        recorded_gates
    ) and all(value is True for value in recorded_gates.values())
    if not checks["all_recorded_builder_gates_passed"]:
        errors.append("one or more recorded builder gates did not pass")

    passed = all(checks.values()) and not errors
    return {
        "grouped_split_verification_schema_version": 1,
        "passed": passed,
        "checks": checks,
        "errors": errors,
        "observed": {
            "outer_folds": outer_folds,
            "assignment_count": len(assignments),
            "total_lineages": total_lineages,
            "assignment_fold_counts": {
                str(fold): assignment_counts.get(fold, 0) for fold in range(outer_folds)
            },
        },
    }


def render_grouped_split_verification_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact independent verification report."""

    observed = report["observed"]
    lines = [
        "# Grouped Split Verification",
        "",
        f"**Status:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Observed",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Outer folds | {observed['outer_folds']} |",
        f"| Assignment count | {observed['assignment_count']:,} |",
        f"| Total lineages | {observed['total_lineages']:,} |",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["checks"].items()
    )
    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    errors.append(f"{name} is missing or is not an object")
    return {}


def _sequence(value: Any, name: str, errors: list[str]) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    errors.append(f"{name} is missing or is not an array")
    return ()


def _integer(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
