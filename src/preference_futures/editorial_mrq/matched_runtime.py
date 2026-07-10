"""Prepare and run Step 8.6 matched generic controls."""

from __future__ import annotations

import gc
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq import transfer as transfer_runtime
from preference_futures.editorial_mrq.matched_common import (
    ARMS,
    BASE_ARM,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXTENDED_L2_GRID,
    PCA_TARGET,
    SCHEMA_VERSION,
    load_contract,
    load_report,
    parse_arms,
    render_plan,
    source,
)
from preference_futures.editorial_mrq.runtime import partition_row_indices
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
)
from preference_futures.training.runtime import _resolve_device, _set_seed


def prepare(
    transfer_directory: Path,
    *,
    output_directory: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    transfer_root = transfer_directory.expanduser().resolve()
    transfer_contract = transfer_runtime._load_transfer_contract(transfer_root)
    output = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else transfer_root / "matched-controls"
    )
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"Step 8.6 output is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    aggregate_path = transfer_root / "aggregate.json"
    audit_path = transfer_root / "specificity-audit.json"
    aggregate = load_report(aggregate_path)
    audit = load_report(audit_path)
    if aggregate.get("future_transfer", {}).get("supported") is not True:
        raise ValueError("Step 8.6 requires the frozen Step 8.4 transfer rule to pass")
    gate = audit.get("interpretation_gate", {})
    if gate.get("dimension_matched_controls_required") is not True:
        raise ValueError("Step 8.6 requires dimension-matched controls")
    if gate.get("extended_l2_diagnostic_indicated") is not True:
        raise ValueError("Step 8.6 requires the extended-L2 diagnostic")

    target_dimensions = {
        arm: int(audit["arms"][PCA_TARGET[arm]]["representation_size"])
        for arm in PCA_TARGET
    }
    contract: dict[str, Any] = {
        "step_8_matched_control_contract_schema_version": SCHEMA_VERSION,
        "status": "frozen_before_matched_control_training",
        "exploratory": True,
        "seed": int(transfer_contract["seed"]),
        "outer_folds": int(transfer_contract["outer_folds"]),
        "arms": list(ARMS),
        "sources": {
            "transfer_contract": source(transfer_root / "contract.json"),
            "transfer_aggregate": source(aggregate_path),
            "specificity_audit": source(audit_path),
        },
        "pca": {
            "fit_partition": "train_only",
            "algorithm": "torch.pca_lowrank",
            "target_dimensions": target_dimensions,
            "oversample": 16,
            "power_iterations": 4,
            "future_labels_used": False,
        },
        "probe": {
            "architecture": "identical Step 8.4 linear probe",
            "l2_grid": list(EXTENDED_L2_GRID),
            "selection_partition": "validation_only",
            "selection_metric": "validation_log_loss",
        },
        "decision_rule": {
            "supported_when": [
                "mrq_choice_aware beats pca_generic_choice_aware with CI below zero",
                "mrq_choice_aware beats extended_generic_choice_aware with CI below zero",
            ],
            "authentic_preference_specificity_claim_made": False,
            "remaining_control": "identically shaped shuffled-preference MR.Q",
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "output_directory": str(output),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    write_json(output / "contract.json", contract)
    (output / "plan.md").write_text(render_plan(contract), encoding="utf-8")
    return contract


def run(
    matched_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    torch, load_file, save_file = transfer_runtime._require_probe_stack()
    root = matched_directory.expanduser().resolve()
    contract = load_contract(root)
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_arms = parse_arms(arms)
    resolved_device = _resolve_device(torch, device)

    transfer_root = Path(contract["sources"]["transfer_contract"]["path"]).parent
    transfer_contract = transfer_runtime._load_transfer_contract(transfer_root)
    embedding = transfer_contract["sources"]["embeddings"]
    tensor_path = Path(str(embedding["tensor_path"]))
    rows_path = Path(str(embedding["rows_path"]))
    transfer_runtime._require_hash(tensor_path, str(embedding["tensor_sha256"]), "embeddings")
    transfer_runtime._require_hash(rows_path, str(embedding["rows_sha256"]), "embedding rows")
    tensors = load_file(str(tensor_path), device="cpu")
    rows = load_jsonl(rows_path)
    context = tensors["context"].float().contiguous()
    candidate_a = tensors["candidate_a"].float().contiguous()
    candidate_b = tensors["candidate_b"].float().contiguous()
    generic = transfer_runtime.build_generic_representations(
        torch, rows, context, candidate_a, candidate_b
    )

    split = transfer_contract["sources"]["split_manifest"]
    split_path = Path(str(split["path"]))
    transfer_runtime._require_hash(split_path, str(split["sha256"]), "split manifest")
    assignments = load_json(split_path).get("lineage_to_outer_fold")
    if not isinstance(assignments, Mapping):
        raise ValueError("split manifest has no lineage assignments")
    labels = [int(bool(row["future_revised"])) for row in rows]

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fold in selected_folds:
        partitions = partition_row_indices(
            rows,
            assignments,
            fold=fold,
            outer_folds=int(contract["outer_folds"]),
        )
        for arm in selected_arms:
            output = root / "runs" / f"fold-{fold:02d}" / arm
            if (output / "report.json").exists() and not force:
                skipped.append({"fold": fold, "arm": arm})
                continue
            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)

            matrix = generic[BASE_ARM[arm]]
            pca_report = None
            if arm in PCA_TARGET:
                matrix, pca_report = train_only_pca(
                    torch=torch,
                    save_file=save_file,
                    matrix=matrix,
                    train_indices=partitions["train"],
                    target_dimensions=int(contract["pca"]["target_dimensions"][arm]),
                    seed=int(contract["seed"]) + fold * 101 + ARMS.index(arm),
                    device=resolved_device,
                    output=output / "pca.safetensors",
                    oversample=int(contract["pca"]["oversample"]),
                    power_iterations=int(contract["pca"]["power_iterations"]),
                )

            print(f"Training Step 8.6 fold {fold:02d} / {arm} ...", flush=True)
            original_grid = transfer_runtime.L2_GRID
            transfer_runtime.L2_GRID = tuple(float(value) for value in contract["probe"]["l2_grid"])
            try:
                report = transfer_runtime._train_future_probe(
                    torch=torch,
                    save_file=save_file,
                    contract=contract,
                    matrix=matrix,
                    rows=rows,
                    labels=labels,
                    partitions=partitions,
                    output=output,
                    fold=fold,
                    arm=arm,
                    device=resolved_device,
                )
            finally:
                transfer_runtime.L2_GRID = original_grid
            report["step_8_matched_control_run_schema_version"] = SCHEMA_VERSION
            report["base_generic_arm"] = BASE_ARM[arm]
            report["l2_grid"] = list(contract["probe"]["l2_grid"])
            report["selected_maximum_l2"] = float(report["selected_l2_lambda"]) == max(
                contract["probe"]["l2_grid"]
            )
            report["pca"] = pca_report
            report.pop("report_sha256", None)
            report["report_sha256"] = canonical_json_sha256(report)
            write_json(output / "report.json", report)
            completed.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "test_log_loss": report["test"]["log_loss"],
                    "selected_l2_lambda": report["selected_l2_lambda"],
                }
            )
            if arm in PCA_TARGET:
                del matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "completed": completed,
        "skipped": skipped,
    }
    write_json(root / "last-run-summary.json", summary)
    return summary


def train_only_pca(
    *,
    torch: Any,
    save_file: Any,
    matrix: Any,
    train_indices: Sequence[int],
    target_dimensions: int,
    seed: int,
    device: Any,
    output: Path,
    oversample: int,
    power_iterations: int,
    batch_size: int = 1024,
) -> tuple[Any, dict[str, Any]]:
    train = matrix.index_select(0, torch.tensor(train_indices, dtype=torch.long)).to(device)
    mean = train.mean(dim=0)
    centered = train - mean
    q = min(target_dimensions + oversample, int(centered.shape[0]), int(centered.shape[1]))
    if q < target_dimensions:
        raise ValueError("PCA rank is below the frozen target dimension")
    _set_seed(torch, seed)
    _, singular_values, components = torch.pca_lowrank(
        centered,
        q=q,
        center=False,
        niter=power_iterations,
    )
    components = components[:, :target_dimensions].contiguous()
    singular_values = singular_values[:target_dimensions].contiguous()
    explained_ratio = float((singular_values.square().sum() / centered.square().sum()).cpu())
    transformed = []
    for start in range(0, int(matrix.shape[0]), batch_size):
        batch = matrix[start : start + batch_size].to(device)
        transformed.append(((batch - mean) @ components).cpu())
    result = torch.cat(transformed, dim=0).float().contiguous()
    save_file(
        {
            "mean": mean.detach().cpu().float(),
            "components": components.detach().cpu().float(),
            "singular_values": singular_values.detach().cpu().float(),
        },
        str(output),
        metadata={"fit_partition": "train_only", "seed": str(seed)},
    )
    return result, {
        "algorithm": "torch.pca_lowrank",
        "fit_partition": "train_only",
        "input_dimensions": int(matrix.shape[1]),
        "target_dimensions": target_dimensions,
        "explained_variance_ratio": explained_ratio,
        "artifact_path": str(output),
        "artifact_sha256": sha256_file(output),
    }
