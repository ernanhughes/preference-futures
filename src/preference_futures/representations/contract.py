"""Freeze the Step 5 representation extraction contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.representations.common import (
    PARTITIONS,
    REPRESENTATION_CONTRACT_SCHEMA_VERSION,
    validate_embedded_hash,
)
from preference_futures.selection.diagnostics import ALL_ARMS
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    positive_int,
    sha256_directory,
    sha256_file,
    write_json,
)
from preference_futures.training.contract import validate_training_contract


def build_representation_contract(
    *,
    selection_manifest_path: Path,
    training_directory: Path,
    output_directory: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Validate Steps 1-4 and freeze all Step 5 extraction choices."""

    selection_path = selection_manifest_path.expanduser().resolve()
    training = training_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    batch_size = positive_int(batch_size, "batch_size")

    training_contract_path = training / "contract.json"
    training_contract = load_json(training_contract_path)
    validate_training_contract(training_contract)
    selection = load_json(selection_path)
    validate_embedded_hash(
        selection,
        hash_field="manifest_sha256",
        label="Step 4 encoder selection manifest",
    )
    if selection.get("status") != "frozen_for_step_5":
        raise ValueError("Step 4 encoder selection manifest is not frozen for Step 5")
    if selection.get("contract_sha256") != training_contract.get("contract_sha256"):
        raise ValueError("Step 4 and Step 3 contract hashes do not match")

    outer_folds = positive_int(training_contract.get("outer_folds"), "outer_folds")
    expected_keys = {(fold, arm) for fold in range(outer_folds) for arm in ALL_ARMS}
    jobs: list[dict[str, Any]] = []
    observed_keys: set[tuple[int, str]] = set()
    for raw_entry in selection.get("entries", []):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Step 4 selection entry must be an object")
        fold = int(raw_entry.get("fold", -1))
        regime = str(raw_entry.get("regime", ""))
        key = (fold, regime)
        if key in observed_keys:
            raise ValueError(f"duplicate Step 4 selection entry: {key}")
        observed_keys.add(key)
        if raw_entry.get("artifact_valid") is not True:
            raise ValueError(f"Step 4 selected an invalid artifact: {key}")
        if raw_entry.get("eligible_for_downstream") is not True:
            raise ValueError(f"Step 4 entry is not eligible for downstream use: {key}")
        encoder_path = Path(str(raw_entry.get("encoder_path", ""))).expanduser().resolve()
        encoder_sha256 = str(raw_entry.get("encoder_sha256", ""))
        if not encoder_path.is_dir() or sha256_directory(encoder_path) != encoder_sha256:
            raise ValueError(f"Step 4 encoder changed or is missing: {key}")
        jobs.append(
            {
                "fold": fold,
                "regime": regime,
                "arm_kind": str(raw_entry.get("arm_kind", "")),
                "encoder_path": str(encoder_path),
                "encoder_sha256": encoder_sha256,
                "source_task_status": raw_entry.get("source_task_status"),
            }
        )
    if observed_keys != expected_keys:
        missing = sorted(expected_keys.difference(observed_keys))
        extra = sorted(observed_keys.difference(expected_keys))
        raise ValueError(f"Step 4 arm coverage mismatch; missing={missing}, extra={extra}")

    step_2_manifest_source = training_contract["sources"]["step_2_manifest"]
    step_2_manifest_path = Path(str(step_2_manifest_source["path"])).expanduser().resolve()
    if sha256_file(step_2_manifest_path) != str(step_2_manifest_source["sha256"]):
        raise ValueError("Step 2 manifest changed after Step 3")
    step_2_manifest = load_json(step_2_manifest_path)
    split_source = step_2_manifest["sources"]["split_manifest"]
    split_manifest_path = Path(str(split_source["path"])).expanduser().resolve()
    if sha256_file(split_manifest_path) != str(split_source["sha256"]):
        raise ValueError("Step 1 split manifest changed after Step 2")

    episodes_source = training_contract["sources"]["episodes"]
    episodes_path = Path(str(episodes_source["path"])).expanduser().resolve()
    if sha256_file(episodes_path) != str(episodes_source["sha256"]):
        raise ValueError("episode source changed after Step 3")
    temporal_source = training_contract["sources"]["temporal_pairs"]
    temporal_pairs_path = Path(str(temporal_source["path"])).expanduser().resolve()
    if sha256_file(temporal_pairs_path) != str(temporal_source["sha256"]):
        raise ValueError("temporal-pair source changed after Step 3")

    snapshot = Path(str(training_contract["model"]["base_snapshot_path"])).expanduser().resolve()
    tokenizer_path = snapshot / "tokenizer"
    tokenizer_sha256 = sha256_directory(tokenizer_path)
    maximum_length = positive_int(
        training_contract["optimisation"]["maximum_sequence_length"],
        "maximum_sequence_length",
    )

    contract: dict[str, Any] = {
        "representation_contract_schema_version": REPRESENTATION_CONTRACT_SCHEMA_VERSION,
        "status": "frozen_before_extraction",
        "seed": int(training_contract["seed"]),
        "outer_folds": outer_folds,
        "arms": list(ALL_ARMS),
        "partitions": list(PARTITIONS),
        "expected_extraction_jobs": len(jobs),
        "expected_partition_artifacts": len(jobs) * len(PARTITIONS),
        "sources": {
            "selection_manifest": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "canonical_sha256": selection["manifest_sha256"],
            },
            "training_contract": {
                "path": str(training_contract_path),
                "sha256": sha256_file(training_contract_path),
                "canonical_sha256": training_contract["contract_sha256"],
            },
            "step_2_manifest": {
                "path": str(step_2_manifest_path),
                "sha256": sha256_file(step_2_manifest_path),
            },
            "split_manifest": {
                "path": str(split_manifest_path),
                "sha256": sha256_file(split_manifest_path),
            },
            "episodes": {
                "path": str(episodes_path),
                "sha256": sha256_file(episodes_path),
            },
            "temporal_pairs": {
                "path": str(temporal_pairs_path),
                "sha256": sha256_file(temporal_pairs_path),
            },
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": tokenizer_sha256,
            "class": training_contract["model"].get("tokenizer_class"),
        },
        "representation": {
            "input_view": "canonical_episode_pair_with_context",
            "input_fields": [
                "context_before",
                "candidate_a",
                "candidate_b",
                "context_after",
            ],
            "serialization": (
                "[CONTEXT_BEFORE] context_before [CANDIDATE_A] candidate_a "
                "[CANDIDATE_B] candidate_b [CONTEXT_AFTER] context_after"
            ),
            "candidate_order": "frozen deterministic Step 1 presentation order",
            "selected_index_exposed_to_encoder": False,
            "future_label_exposed_to_encoder": False,
            "v2_fields_exposed_to_encoder": False,
            "maximum_sequence_length": maximum_length,
            "padding": "max_length",
            "truncation": True,
            "pooling": "final_hidden_state_first_token",
            "pooling_token": "[CLS]",
            "output_dtype": "float32",
            "batch_size": batch_size,
            "model_mode": "eval",
            "gradient_tracking": False,
            "row_order": "episode_id ascending within each partition",
            "row_metadata": ["row_index", "episode_id", "lineage_id", "input_sha256"],
        },
        "jobs": sorted(jobs, key=lambda item: (int(item["fold"]), str(item["regime"]))),
        "gates": {
            "step_3_contract_valid": True,
            "step_4_manifest_hash_valid": True,
            "all_seven_arms_present_per_fold": True,
            "all_encoders_mechanically_valid": True,
            "split_manifest_hash_matches": True,
            "episode_hash_matches": True,
            "future_fields_forbidden_from_encoder_input": True,
            "one_pooling_rule_for_all_arms": True,
            "one_tokenizer_for_all_arms": True,
            "one_maximum_length_for_all_arms": True,
            "source_task_heads_not_loaded": True,
        },
        "warnings": [
            (
                "Step 5 extracts representations only. It does not train, calibrate or select a "
                "future-outcome probe."
            ),
            (
                "The final-layer first-token state is frozen because DistilBERT sequence "
                "classification uses that state before its task-specific head."
            ),
            (
                "Rows contain identifiers and input hashes only. Future labels are joined from the "
                "frozen episode source in Step 6."
            ),
        ],
        "output_directory": str(output),
    }
    if not all(contract["gates"].values()):
        raise ValueError("one or more Step 5 contract gates failed")
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def validate_representation_contract(contract: Mapping[str, Any]) -> None:
    validate_embedded_hash(
        contract,
        hash_field="contract_sha256",
        label="Step 5 representation contract",
    )
    if contract.get("status") != "frozen_before_extraction":
        raise ValueError("Step 5 representation contract has an invalid status")
    if tuple(contract.get("arms", ())) != tuple(ALL_ARMS):
        raise ValueError("Step 5 arm order changed")
    if tuple(contract.get("partitions", ())) != PARTITIONS:
        raise ValueError("Step 5 partition order changed")
    gates = contract.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("one or more Step 5 contract gates failed")
    for source in contract.get("sources", {}).values():
        path = Path(str(source.get("path", ""))).expanduser().resolve()
        expected = str(source.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Step 5 source changed: {path}")
    tokenizer = contract.get("tokenizer", {})
    tokenizer_path = Path(str(tokenizer.get("path", ""))).expanduser().resolve()
    if sha256_directory(tokenizer_path) != str(tokenizer.get("sha256", "")):
        raise ValueError("Step 5 tokenizer changed")
    for job in contract.get("jobs", []):
        encoder = Path(str(job.get("encoder_path", ""))).expanduser().resolve()
        if sha256_directory(encoder) != str(job.get("encoder_sha256", "")):
            raise ValueError(
                f"Step 5 encoder changed: fold {job.get('fold')} {job.get('regime')}"
            )


def write_representation_contract(
    output_directory: Path,
    contract: Mapping[str, Any],
) -> None:
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    (output / "extraction-plan.md").write_text(
        render_representation_plan_markdown(contract),
        encoding="utf-8",
    )


def render_representation_plan_markdown(contract: Mapping[str, Any]) -> str:
    representation = contract["representation"]
    lines = [
        "# Frozen Representation Extraction Plan",
        "",
        "## Contract",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Outer folds | {contract['outer_folds']} |",
        f"| Arms per fold | {len(contract['arms'])} |",
        f"| Extraction jobs | {contract['expected_extraction_jobs']} |",
        f"| Partition matrices | {contract['expected_partition_artifacts']} |",
        f"| Batch size | {representation['batch_size']} |",
        f"| Maximum sequence length | {representation['maximum_sequence_length']} |",
        f"| Pooling | `{representation['pooling']}` |",
        f"| Output dtype | `{representation['output_dtype']}` |",
        "",
        "## Leakage boundary",
        "",
        "- Candidate order remains the frozen Step 1 order.",
        "- The encoder receives no selected-index label.",
        "- The encoder receives no future label or V2 field.",
        "- Output row metadata contains identifiers and input hashes only.",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in contract["gates"].items()
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in contract["warnings"])
    lines.append("")
    return "\n".join(lines)
