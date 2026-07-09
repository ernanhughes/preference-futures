"""Independent verification for persisted Step 5 representation matrices."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.representations.common import (
    FORBIDDEN_ROW_KEYS,
    PARTITIONS,
    REPRESENTATION_VERIFICATION_SCHEMA_VERSION,
    parse_arm_selection,
)
from preference_futures.representations.contract import validate_representation_contract
from preference_futures.representations.runtime import _partition_indices
from preference_futures.training.common import (
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
)
from preference_futures.training.data import load_source_store, serialise_episode
from preference_futures.training.runtime import _require_training_stack


def verify_representation_runs(
    representation_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
) -> dict[str, Any]:
    """Verify persisted Step 5 artifacts without recomputing model outputs."""

    stack = _require_training_stack()
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Step 5 verification requires safetensors. Install with: "
            "python -m pip install -e '.[train]'"
        ) from exc
    torch = stack["torch"]

    root = representation_directory.expanduser().resolve()
    contract = load_json(root / "contract.json")
    errors: list[str] = []
    try:
        validate_representation_contract(contract)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"representation contract is invalid: {exc}")

    outer_folds = int(contract.get("outer_folds", 0))
    selected_folds = parse_int_selection(folds, upper_bound=outer_folds)
    selected_arms = parse_arm_selection(arms)
    expected_keys = {(fold, arm) for fold in selected_folds for arm in selected_arms}
    jobs = {
        (int(job["fold"]), str(job["regime"])): job
        for job in contract.get("jobs", [])
        if (int(job["fold"]), str(job["regime"])) in expected_keys
    }
    if set(jobs) != expected_keys:
        errors.append("selected Step 5 jobs do not match the frozen contract")

    source_store = load_source_store(
        Path(contract["sources"]["episodes"]["path"]),
        Path(contract["sources"]["temporal_pairs"]["path"]),
    )
    split_manifest = load_json(Path(contract["sources"]["split_manifest"]["path"]))
    episode_ids = sorted(source_store.episodes)
    episodes = [source_store.episodes[episode_id] for episode_id in episode_ids]
    input_hashes = [
        hashlib.sha256(serialise_episode(episode).encode("utf-8")).hexdigest()
        for episode in episodes
    ]
    partition_indices = _partition_indices(
        episodes,
        split_manifest,
        outer_folds=outer_folds,
    )

    observed_keys: set[tuple[int, str]] = set()
    row_signatures: dict[tuple[int, str], set[str]] = defaultdict(set)
    hidden_sizes: set[int] = set()
    device_types: set[str] = set()
    runtime_environments: set[tuple[str, ...]] = set()
    run_summaries: list[dict[str, Any]] = []
    test_episode_counts: Counter[str] = Counter()

    for key in sorted(expected_keys):
        fold, regime = key
        run_directory = root / "runs" / f"fold-{fold:02d}" / regime
        report_path = run_directory / "run.json"
        if not report_path.exists():
            errors.append(f"missing Step 5 run report: fold {fold} {regime}")
            continue
        try:
            report = load_json(report_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid Step 5 run report: fold {fold} {regime}: {exc}")
            continue
        observed_keys.add(key)
        job = jobs[key]
        if report.get("status") != "complete":
            errors.append(f"incomplete Step 5 run: fold {fold} {regime}")
        if report.get("contract_sha256") != contract.get("contract_sha256"):
            errors.append(f"Step 5 contract hash mismatch: fold {fold} {regime}")
        if report.get("encoder_sha256") != job.get("encoder_sha256"):
            errors.append(f"Step 5 encoder hash mismatch: fold {fold} {regime}")
        representation = report.get("representation", {})
        if representation.get("future_fields_exposed") is not False:
            errors.append(f"future fields exposed: fold {fold} {regime}")
        if representation.get("selected_index_exposed") is not False:
            errors.append(f"selected index exposed: fold {fold} {regime}")
        hidden_size = int(representation.get("hidden_size", 0))
        if hidden_size < 1:
            errors.append(f"invalid hidden size: fold {fold} {regime}")
        else:
            hidden_sizes.add(hidden_size)
        environment = report.get("environment", {})
        device_types.add(str(environment.get("device", "")).split(":", maxsplit=1)[0])
        runtime_environments.add(
            (
                str(environment.get("python", "")),
                str(environment.get("torch", "")),
                str(environment.get("transformers", "")),
                str(environment.get("device_name", "")),
            )
        )

        partition_summary: dict[str, int] = {}
        for partition in PARTITIONS:
            expected_indices = list(partition_indices[fold][partition])
            expected_rows = [
                {
                    "row_index": row_index,
                    "episode_id": str(episodes[source_index]["episode_id"]),
                    "lineage_id": str(episodes[source_index]["lineage_id"]),
                    "input_sha256": input_hashes[source_index],
                }
                for row_index, source_index in enumerate(expected_indices)
            ]
            vector_path = run_directory / f"{partition}.safetensors"
            rows_path = run_directory / f"{partition}.rows.jsonl"
            artifact = report.get("artifacts", {}).get(partition, {})
            if not vector_path.exists() or not rows_path.exists():
                errors.append(f"missing partition artifacts: fold {fold} {regime} {partition}")
                continue
            if sha256_file(vector_path) != artifact.get("representations_sha256"):
                errors.append(
                    f"representation hash mismatch: fold {fold} {regime} {partition}"
                )
            if sha256_file(rows_path) != artifact.get("rows_sha256"):
                errors.append(f"row hash mismatch: fold {fold} {regime} {partition}")

            rows = load_jsonl(rows_path)
            if rows != expected_rows:
                errors.append(f"row identity mismatch: fold {fold} {regime} {partition}")
            if any(FORBIDDEN_ROW_KEYS.intersection(row) for row in rows):
                errors.append(f"forbidden row metadata: fold {fold} {regime} {partition}")
            row_signatures[(fold, partition)].add(sha256_file(rows_path))
            if partition == "test" and regime == selected_arms[0]:
                test_episode_counts.update(str(row["episode_id"]) for row in rows)

            tensors = load_file(str(vector_path), device="cpu")
            if set(tensors) != {"representations"}:
                errors.append(f"unexpected tensor keys: fold {fold} {regime} {partition}")
                continue
            matrix = tensors["representations"]
            expected_shape = (len(expected_rows), hidden_size)
            if tuple(matrix.shape) != expected_shape:
                errors.append(
                    f"matrix shape mismatch: fold {fold} {regime} {partition}: "
                    f"{tuple(matrix.shape)} != {expected_shape}"
                )
            if matrix.dtype != torch.float32:
                errors.append(f"matrix dtype mismatch: fold {fold} {regime} {partition}")
            if not bool(torch.isfinite(matrix).all().item()):
                errors.append(f"non-finite matrix: fold {fold} {regime} {partition}")
            if artifact.get("shape") != [len(expected_rows), hidden_size]:
                errors.append(f"reported shape mismatch: fold {fold} {regime} {partition}")
            if artifact.get("dtype") != "float32":
                errors.append(f"reported dtype mismatch: fold {fold} {regime} {partition}")
            partition_summary[partition] = len(expected_rows)

        run_summaries.append(
            {
                "fold": fold,
                "regime": regime,
                "encoder_sha256": str(job["encoder_sha256"]),
                "hidden_size": hidden_size,
                "partition_counts": partition_summary,
            }
        )

    for (fold, partition), signatures in row_signatures.items():
        if len(signatures) != 1:
            errors.append(f"row ordering differs across arms: fold {fold} {partition}")
    if len(hidden_sizes) != 1:
        errors.append(f"expected one hidden size, observed {sorted(hidden_sizes)}")
    if len(device_types) != 1:
        errors.append(f"expected one device type, observed {sorted(device_types)}")
    if len(runtime_environments) != 1:
        errors.append("expected one Step 5 runtime environment")
    if selected_folds == tuple(range(outer_folds)):
        expected_test_ids = set(episode_ids)
        if set(test_episode_counts) != expected_test_ids or any(
            count != 1 for count in test_episode_counts.values()
        ):
            errors.append("outer test partitions do not cover every episode exactly once")

    checks = {
        "representation_contract_is_valid": not any(
            error.startswith("representation contract is invalid") for error in errors
        ),
        "all_expected_runs_exist": observed_keys == expected_keys,
        "all_run_statuses_are_complete": not any(
            error.startswith("incomplete Step 5 run") for error in errors
        ),
        "contract_and_encoder_hashes_match": not any(
            "contract hash mismatch" in error or "encoder hash mismatch" in error
            for error in errors
        ),
        "all_partition_artifact_hashes_match": not any(
            "hash mismatch" in error and "contract" not in error and "encoder" not in error
            for error in errors
        ),
        "row_identities_match_frozen_partitions": not any(
            error.startswith("row identity mismatch") for error in errors
        ),
        "row_metadata_contains_no_labels": not any(
            error.startswith("forbidden row metadata") for error in errors
        ),
        "representations_are_finite_float32": not any(
            error.startswith("non-finite matrix") or error.startswith("matrix dtype mismatch")
            for error in errors
        ),
        "representation_shapes_match": not any(
            "shape mismatch" in error for error in errors
        ),
        "row_order_is_identical_across_arms": not any(
            error.startswith("row ordering differs") for error in errors
        ),
        "one_hidden_size_used": len(hidden_sizes) == 1,
        "one_device_type_used": len(device_types) == 1,
        "one_runtime_environment_used": len(runtime_environments) == 1,
        "outer_test_coverage_is_exact": not any(
            error.startswith("outer test partitions") for error in errors
        ),
    }
    report = {
        "representation_verification_schema_version": REPRESENTATION_VERIFICATION_SCHEMA_VERSION,
        "status": "pass" if not errors and all(checks.values()) else "fail",
        "passed": not errors and all(checks.values()),
        "contract_sha256": contract.get("contract_sha256"),
        "selection": {"folds": list(selected_folds), "arms": list(selected_arms)},
        "observed": {
            "expected_jobs": len(expected_keys),
            "observed_jobs": len(observed_keys),
            "partition_artifacts": len(observed_keys) * len(PARTITIONS),
            "hidden_sizes": sorted(hidden_sizes),
            "device_types": sorted(device_types),
            "runtime_environment_count": len(runtime_environments),
        },
        "checks": checks,
        "errors": errors,
        "runs": run_summaries,
    }
    return report


def write_representation_verification(
    representation_directory: Path,
    report: Mapping[str, Any],
) -> None:
    root = representation_directory.expanduser().resolve()
    write_json(root / "representation-verification.json", report)
    (root / "representation-verification.md").write_text(
        render_representation_verification_markdown(report),
        encoding="utf-8",
    )


def render_representation_verification_markdown(report: Mapping[str, Any]) -> str:
    observed = report["observed"]
    lines = [
        "# Frozen Representation Verification",
        "",
        f"**Status:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Observed",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Expected jobs | {observed['expected_jobs']} |",
        f"| Observed jobs | {observed['observed_jobs']} |",
        f"| Partition matrices | {observed['partition_artifacts']} |",
        f"| Hidden sizes | {', '.join(map(str, observed['hidden_sizes']))} |",
        f"| Device types | {', '.join(observed['device_types'])} |",
        f"| Runtime environments | {observed['runtime_environment_count']} |",
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
