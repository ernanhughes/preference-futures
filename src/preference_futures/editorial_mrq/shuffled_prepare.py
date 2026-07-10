"""Freeze the final shuffled-preference MR.Q specificity contract."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq.shuffled_common import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    REQUIRED_NEGATIVE_REPLICATES,
    SHUFFLED_ARMS,
    SHUFFLED_CONTROL_SCHEMA_VERSION,
    SHUFFLE_SEEDS,
    load_canonical_report,
)
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)


def prepare_shuffled_control(
    editorial_directory: Path,
    *,
    output_directory: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze Step 8.7 after the matched-control result is available."""

    editorial_root = editorial_directory.expanduser().resolve()
    transfer_root = editorial_root / "future-transfer"
    output = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else transfer_root / "shuffled-mrq-control"
    )
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"Step 8.7 output is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    editorial_contract_path = editorial_root / "contract.json"
    editorial_contract = load_json(editorial_contract_path)
    transfer_contract_path = transfer_root / "contract.json"
    transfer_contract = load_json(transfer_contract_path)
    transfer_aggregate_path = transfer_root / "aggregate.json"
    transfer_aggregate = load_canonical_report(transfer_aggregate_path)
    matched_aggregate_path = transfer_root / "matched-controls" / "aggregate.json"
    matched_aggregate = load_canonical_report(matched_aggregate_path)
    if transfer_aggregate.get("future_transfer", {}).get("supported") is not True:
        raise ValueError("Step 8.7 requires the completed positive Step 8.4 result")
    specificity = matched_aggregate.get("compression_and_regularisation_specificity", {})
    if specificity.get("authentic_preference_specificity_claim_made") is not False:
        raise ValueError("Step 8.7 expected authentic preference specificity to remain unclaimed")

    embedding_report_path = editorial_root / "embeddings" / "report.json"
    embedding_report = load_json(embedding_report_path)
    tensor_path = Path(str(embedding_report["artifacts"]["embeddings_path"]))
    rows_path = Path(str(embedding_report["artifacts"]["rows_path"]))
    split_path = Path(str(editorial_contract["sources"]["split_manifest"]["path"]))
    for path, expected, label in (
        (tensor_path, embedding_report["artifacts"]["embeddings_sha256"], "embeddings"),
        (rows_path, embedding_report["artifacts"]["rows_sha256"], "embedding rows"),
        (split_path, editorial_contract["sources"]["split_manifest"]["sha256"], "split manifest"),
    ):
        if sha256_file(path) != str(expected):
            raise ValueError(f"Step 8.7 {label} changed")

    contract: dict[str, Any] = {
        "step_8_shuffled_control_schema_version": SHUFFLED_CONTROL_SCHEMA_VERSION,
        "status": "frozen_before_shuffled_source_training",
        "exploratory": True,
        "seed": int(editorial_contract["seed"]),
        "outer_folds": int(editorial_contract["outer_folds"]),
        "shuffle_replicates": len(SHUFFLE_SEEDS),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "arms": list(SHUFFLED_ARMS),
        "sources": {
            "editorial_contract": _file_source(editorial_contract_path),
            "transfer_contract": _file_source(transfer_contract_path),
            "transfer_aggregate": _file_source(transfer_aggregate_path),
            "matched_aggregate": _file_source(matched_aggregate_path),
            "embedding_report": _file_source(embedding_report_path),
            "embedding_tensor": _file_source(tensor_path),
            "embedding_rows": _file_source(rows_path),
            "split_manifest": _file_source(split_path),
        },
        "source_control": {
            "architecture": "identical EditorialMRQ network",
            "embedder": "same frozen MPNet embeddings",
            "training_settings": dict(editorial_contract["ranker"]),
            "teacher_predictions": "none",
            "label_null": (
                "selected_index labels permuted independently within train, validation, and test "
                "partitions for each fold and replicate"
            ),
            "class_counts_preserved_within_partition": True,
            "checkpoint_selection": "shuffled validation log loss",
            "authentic_preference_labels_used_for_source_training_or_selection": False,
            "authentic_source_test_labels_used_for_diagnostic_only_after_checkpoint_freeze": True,
        },
        "downstream": {
            "representations": {
                "shuffled_mrq_blind": "same 129-dimensional MR.Q blind state",
                "shuffled_mrq_choice_aware": (
                    "same 193-dimensional MR.Q state oriented by authentic selected/rejected choice"
                ),
            },
            "future_probe": dict(transfer_contract["probe"]),
            "future_labels_hidden_until_shuffled_source_model_is_frozen": True,
            "authentic_historical_choice_used_for_choice_aware_orientation": True,
        },
        "estimand": {
            "primary": (
                "authentic mrq_choice_aware loss minus mean shuffled_mrq_choice_aware loss"
            ),
            "secondary": "authentic mrq_blind loss minus mean shuffled_mrq_blind loss",
            "negative_value_means": "authentic preference supervision improves compact transfer",
            "unit": "pooled out-of-fold episode",
            "uncertainty": "paired article-lineage bootstrap over per-episode loss differences",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "decision_rule": {
            "mean_primary_interval_entirely_below_zero": True,
            "mean_secondary_interval_entirely_below_zero": True,
            "minimum_negative_point_estimate_replicates": REQUIRED_NEGATIVE_REPLICATES,
            "replicates_total": len(SHUFFLE_SEEDS),
            "note": (
                "Authentic preference specificity is supported only if both mean comparisons pass "
                "and authentic MR.Q beats at least four of five shuffled replicas in both arms."
            ),
        },
        "gates": {
            "step_8_4_result_available": True,
            "step_8_6_result_available": True,
            "same_architecture": True,
            "same_folds": True,
            "same_embeddings": True,
            "same_source_training_budget": True,
            "same_downstream_probe": True,
            "future_labels_hidden_during_source_training": True,
        },
        "output_directory": str(output),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    write_json(output / "contract.json", contract)
    (output / "plan.md").write_text(render_plan(contract), encoding="utf-8")
    return contract


def render_plan(contract: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Step 8.7 Editorial MR.Q — Shuffled-Preference Control Plan",
            "",
            f"- Shuffled replicas: `{int(contract['shuffle_replicates'])}`",
            "- Source architecture and training settings: identical to authentic MR.Q",
            "- Label permutation: within fold partition, preserving exact class counts",
            "- Source checkpoint selection: shuffled validation labels only",
            "- Downstream future probes: identical to Step 8.4",
            "",
            "## Primary comparison",
            "",
            "`authentic mrq_choice_aware` versus the mean of five shuffled-label MR.Q controls.",
            "",
            "## Decision rule",
            "",
            str(contract["decision_rule"]["note"]),
            "",
        ]
    )


def _file_source(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}
