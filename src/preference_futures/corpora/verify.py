"""Verify persisted Step 2 corpus artifacts independently of the builder."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.corpora.common import CORPUS_NAMES, FORBIDDEN_FUTURE_KEYS


def verify_compute_matched_corpora(output_directory: Path) -> dict[str, Any]:
    """Verify file coverage, counts, record identity, forbidden fields and source hashes."""

    output = output_directory.expanduser().resolve()
    manifest_path = output / "manifest.json"
    errors: list[str] = []
    checks: dict[str, bool] = {}

    if not manifest_path.exists():
        return {
            "corpus_verification_schema_version": 1,
            "passed": False,
            "checks": {"manifest_exists": False},
            "errors": [f"manifest does not exist: {manifest_path}"],
            "observed": {},
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks["manifest_exists"] = True
    corpus_names = manifest.get("corpora")
    checks["six_expected_corpora_declared"] = corpus_names == list(CORPUS_NAMES)
    if not checks["six_expected_corpora_declared"]:
        errors.append(f"unexpected corpus list: {corpus_names}")

    builder_gates = manifest.get("gates")
    checks["all_builder_gates_passed"] = isinstance(builder_gates, Mapping) and bool(
        builder_gates
    ) and all(value is True for value in builder_gates.values())
    if not checks["all_builder_gates_passed"]:
        errors.append("one or more builder gates are missing or failed")

    source_hashes_match = True
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        source_hashes_match = False
        errors.append("manifest sources are missing")
    else:
        for name, source in sources.items():
            if source is None:
                continue
            if not isinstance(source, Mapping):
                source_hashes_match = False
                errors.append(f"source metadata is invalid: {name}")
                continue
            path_value = source.get("path")
            expected_hash = source.get("sha256")
            if not isinstance(path_value, str) or not isinstance(expected_hash, str):
                source_hashes_match = False
                errors.append(f"source path/hash is missing: {name}")
                continue
            source_path = Path(path_value)
            if not source_path.exists():
                source_hashes_match = False
                errors.append(f"source file no longer exists: {source_path}")
                continue
            observed_hash = _sha256(source_path)
            if observed_hash != expected_hash:
                source_hashes_match = False
                errors.append(
                    f"source hash changed for {name}: {observed_hash} != {expected_hash}"
                )
    checks["source_hashes_match_manifest"] = source_hashes_match

    expected_files = 0
    observed_files = 0
    records_seen = 0
    persisted_counts_match = True
    record_identity_matches_path = True
    no_future_fields = True
    unique_source_ids_within_file = True

    folds = manifest.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes, bytearray)):
        folds = []
        errors.append("manifest folds are missing or invalid")

    for fold_summary in folds:
        if not isinstance(fold_summary, Mapping):
            persisted_counts_match = False
            errors.append("fold summary is not an object")
            continue
        fold = fold_summary.get("fold")
        partitions = fold_summary.get("partitions")
        if not isinstance(fold, int) or isinstance(fold, bool) or not isinstance(
            partitions, Mapping
        ):
            persisted_counts_match = False
            errors.append(f"invalid fold summary: {fold_summary}")
            continue
        for partition in ("train", "validation"):
            summary = partitions.get(partition)
            expected_count = (
                summary.get("records_per_corpus")
                if isinstance(summary, Mapping)
                else None
            )
            if not isinstance(expected_count, int) or isinstance(expected_count, bool):
                persisted_counts_match = False
                errors.append(f"fold {fold} {partition} has no records_per_corpus")
                continue
            for corpus_name in CORPUS_NAMES:
                expected_files += 1
                path = output / f"fold-{fold:02d}" / corpus_name / f"{partition}.jsonl"
                if not path.exists():
                    persisted_counts_match = False
                    errors.append(f"missing corpus file: {path}")
                    continue
                observed_files += 1
                count = 0
                source_ids: set[str] = set()
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        count += 1
                        records_seen += 1
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            persisted_counts_match = False
                            errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                            continue
                        if not isinstance(record, Mapping):
                            record_identity_matches_path = False
                            errors.append(f"{path}:{line_number}: record is not an object")
                            continue
                        if (
                            record.get("corpus") != corpus_name
                            or record.get("fold") != fold
                            or record.get("partition") != partition
                        ):
                            record_identity_matches_path = False
                            errors.append(
                                f"{path}:{line_number}: record identity disagrees with path"
                            )
                        if FORBIDDEN_FUTURE_KEYS.intersection(record):
                            no_future_fields = False
                            errors.append(
                                f"{path}:{line_number}: future field leaked into source task"
                            )
                        source_id = str(record.get("source_id", ""))
                        if not source_id or source_id in source_ids:
                            unique_source_ids_within_file = False
                            errors.append(f"{path}:{line_number}: missing or duplicate source_id")
                        source_ids.add(source_id)
                if count != expected_count:
                    persisted_counts_match = False
                    errors.append(
                        f"{path}: observed {count} records but expected {expected_count}"
                    )

    checks["all_expected_corpus_files_exist"] = (
        expected_files > 0 and observed_files == expected_files
    )
    checks["persisted_record_counts_match_manifest"] = persisted_counts_match
    checks["record_identity_matches_file_path"] = record_identity_matches_path
    checks["no_persisted_source_record_contains_future_fields"] = no_future_fields
    checks["source_ids_unique_within_each_corpus_file"] = unique_source_ids_within_file

    temporal_path = output / "temporal-pairs.jsonl"
    temporal_audit_path = output / "temporal-pairs-audit.json"
    checks["temporal_pool_and_audit_exist"] = (
        temporal_path.exists() and temporal_audit_path.exists()
    )
    if not checks["temporal_pool_and_audit_exist"]:
        errors.append("temporal-pairs.jsonl or temporal-pairs-audit.json is missing")

    passed = all(checks.values()) and not errors
    return {
        "corpus_verification_schema_version": 1,
        "passed": passed,
        "checks": checks,
        "errors": errors,
        "observed": {
            "expected_corpus_files": expected_files,
            "observed_corpus_files": observed_files,
            "records_seen": records_seen,
        },
    }


def render_corpus_verification_markdown(report: Mapping[str, Any]) -> str:
    observed = report["observed"]
    lines = [
        "# Compute-Matched Corpus Verification",
        "",
        f"**Status:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Observed",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Expected corpus files | {observed.get('expected_corpus_files', 0):,} |",
        f"| Observed corpus files | {observed.get('observed_corpus_files', 0):,} |",
        f"| Source-task records read | {observed.get('records_seen', 0):,} |",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
