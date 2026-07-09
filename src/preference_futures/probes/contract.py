"""Freeze the Step 6 identical future-probe contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.probes.common import (
    L2_GRID,
    PROBE_CONTRACT_SCHEMA_VERSION,
    SELECTION_TOLERANCE,
    STANDARDISATION_EPSILON,
    validate_embedded_hash,
)
from preference_futures.representations.common import PARTITIONS
from preference_futures.representations.contract import validate_representation_contract
from preference_futures.selection.diagnostics import ALL_ARMS
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)


def build_probe_contract(
    *,
    representation_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Validate the complete Step 5 archive and freeze Step 6 before outcomes are read."""

    representations = representation_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    representation_contract_path = representations / "contract.json"
    representation_verification_path = representations / "representation-verification.json"
    representation_contract = load_json(representation_contract_path)
    validate_representation_contract(representation_contract)
    verification = load_json(representation_verification_path)
    _require_complete_representation_verification(verification, representation_contract)

    episodes_path = Path(
        str(representation_contract["sources"]["episodes"]["path"])
    ).expanduser().resolve()
    if sha256_file(episodes_path) != str(
        representation_contract["sources"]["episodes"]["sha256"]
    ):
        raise ValueError("Step 6 episode source changed after Step 5")
    episodes = load_jsonl(episodes_path)
    episode_ids: set[str] = set()
    for record in episodes:
        episode_id = str(record.get("episode_id", ""))
        if not episode_id or episode_id in episode_ids:
            raise ValueError("Step 6 episode IDs are missing or duplicated")
        episode_ids.add(episode_id)
        if type(record.get("future_revised")) is not bool:
            raise ValueError(f"episode {episode_id} has no boolean future_revised label")

    jobs: list[dict[str, Any]] = []
    expected_keys = {
        (fold, arm)
        for fold in range(int(representation_contract["outer_folds"]))
        for arm in ALL_ARMS
    }
    observed_keys: set[tuple[int, str]] = set()
    for representation_job in representation_contract["jobs"]:
        fold = int(representation_job["fold"])
        regime = str(representation_job["regime"])
        key = (fold, regime)
        if key in observed_keys:
            raise ValueError(f"duplicate Step 5 representation job: {key}")
        observed_keys.add(key)
        run_directory = representations / "runs" / f"fold-{fold:02d}" / regime
        report_path = run_directory / "run.json"
        report = load_json(report_path)
        if report.get("status") != "complete":
            raise ValueError(f"Step 5 run is incomplete: fold {fold} {regime}")
        if report.get("contract_sha256") != representation_contract.get("contract_sha256"):
            raise ValueError(f"Step 5 contract mismatch: fold {fold} {regime}")
        if report.get("encoder_sha256") != representation_job.get("encoder_sha256"):
            raise ValueError(f"Step 5 encoder mismatch: fold {fold} {regime}")

        artifacts: dict[str, Any] = {}
        for partition in PARTITIONS:
            source = report["artifacts"][partition]
            vector_path = run_directory / str(source["representations_path"])
            rows_path = run_directory / str(source["rows_path"])
            if sha256_file(vector_path) != str(source["representations_sha256"]):
                raise ValueError(
                    f"Step 5 vector changed: fold {fold} {regime} {partition}"
                )
            if sha256_file(rows_path) != str(source["rows_sha256"]):
                raise ValueError(f"Step 5 rows changed: fold {fold} {regime} {partition}")
            artifacts[partition] = {
                "representations_path": str(vector_path),
                "representations_sha256": str(source["representations_sha256"]),
                "rows_path": str(rows_path),
                "rows_sha256": str(source["rows_sha256"]),
                "rows": int(source["rows"]),
                "shape": list(source["shape"]),
                "dtype": str(source["dtype"]),
            }
        jobs.append(
            {
                "fold": fold,
                "regime": regime,
                "encoder_sha256": str(representation_job["encoder_sha256"]),
                "source_task_status": representation_job.get("source_task_status"),
                "representation_run_path": str(report_path),
                "representation_run_sha256": sha256_file(report_path),
                "artifacts": artifacts,
            }
        )
    if observed_keys != expected_keys:
        raise ValueError("Step 5 does not contain exactly seven arms for every fold")

    contract: dict[str, Any] = {
        "probe_contract_schema_version": PROBE_CONTRACT_SCHEMA_VERSION,
        "status": "frozen_before_future_probe_training",
        "seed": int(representation_contract["seed"]),
        "outer_folds": int(representation_contract["outer_folds"]),
        "arms": list(ALL_ARMS),
        "expected_probe_jobs": len(jobs),
        "sources": {
            "representation_contract": {
                "path": str(representation_contract_path),
                "sha256": sha256_file(representation_contract_path),
                "canonical_sha256": representation_contract["contract_sha256"],
            },
            "representation_verification": {
                "path": str(representation_verification_path),
                "sha256": sha256_file(representation_verification_path),
            },
            "episodes": {
                "path": str(episodes_path),
                "sha256": sha256_file(episodes_path),
                "records": len(episodes),
            },
        },
        "target": {
            "field": "future_revised",
            "positive_class": "selected V1 changes in the next observed state",
            "negative_class": "selected V1 remains stable in the next observed state",
            "joined_by": "episode_id after representation extraction",
        },
        "preprocessing": {
            "method": "featurewise_z_score",
            "fit_partition": "train_only",
            "mean": "population mean over train rows",
            "scale": "population standard deviation over train rows",
            "zero_variance_scale": 1.0,
            "epsilon": STANDARDISATION_EPSILON,
            "apply_to": ["train", "validation", "test"],
        },
        "probe": {
            "architecture": "single_linear_logit_with_bias",
            "link": "sigmoid",
            "class_weighting": "none",
            "initialisation": "all weights and bias equal zero",
            "training_partition": "train_only",
            "batching": "full_batch",
            "loss": "mean binary cross entropy with logits plus L2 on weights only",
            "l2_grid": list(L2_GRID),
            "optimizer": "torch.optim.LBFGS",
            "optimizer_settings": {
                "lr": 1.0,
                "max_iter": 100,
                "max_eval": 125,
                "tolerance_grad": 1e-7,
                "tolerance_change": 1e-9,
                "history_size": 10,
                "line_search_fn": "strong_wolfe",
            },
            "selection_partition": "validation_only",
            "selection_metric": "validation_log_loss",
            "selection_direction": "minimise",
            "selection_tolerance": SELECTION_TOLERANCE,
            "tie_break": "stronger_l2",
            "retrain_after_selection": False,
            "calibration": "none; sigmoid of selected logistic probe logit",
            "test_evaluations_per_job": 1,
            "precision": "float32",
        },
        "metrics": {
            "primary": "test_log_loss",
            "secondary": ["test_brier_score", "test_roc_auc"],
            "descriptive": ["test_accuracy", "mean_probability", "prevalence"],
            "constant_baseline": "fold-specific train prevalence",
        },
        "confirmatory_estimand": {
            "primary_comparison": "generic_test_log_loss - authentic_preference_test_log_loss",
            "positive_value_means": "authentic preference improves future prediction",
            "unit": "pooled out-of-fold episode",
            "uncertainty": "paired article-lineage bootstrap",
            "bootstrap_seed": 17,
            "bootstrap_replicates": 10000,
            "confidence_interval": "two-sided 95% percentile",
            "specificity_comparisons": [
                "authentic_preference versus random_label",
                "authentic_preference versus shuffled_preference",
            ],
            "other_descriptive_comparisons": [
                "authentic_preference versus language_adaptation",
                "authentic_preference versus pair_exposure",
                "authentic_preference versus temporal_direction",
            ],
        },
        "jobs": sorted(jobs, key=lambda item: (int(item["fold"]), str(item["regime"]))),
        "gates": {
            "step_5_contract_valid": True,
            "step_5_full_verification_passed": True,
            "all_70_representation_jobs_present": True,
            "all_representation_hashes_match": True,
            "one_linear_architecture_for_all_arms": True,
            "one_preprocessing_rule_for_all_arms": True,
            "one_l2_grid_for_all_arms": True,
            "validation_only_model_selection": True,
            "test_partition_not_used_for_training_selection_or_calibration": True,
            "future_label_joined_only_after_step_5": True,
        },
        "warnings": [
            "Step 6 tests linear decodability, not every possible nonlinear use of a representation.",
            "No class weighting is used because probabilistic forecast loss is the primary metric.",
            "No post-hoc calibration is applied; the logistic link is the frozen probability map.",
            "Do not change the L2 grid or optimizer after inspecting any test prediction.",
        ],
        "output_directory": str(output),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _require_complete_representation_verification(
    verification: Mapping[str, Any],
    representation_contract: Mapping[str, Any],
) -> None:
    if verification.get("passed") is not True or verification.get("status") != "pass":
        raise ValueError("Step 5 full representation verification has not passed")
    expected_folds = list(range(int(representation_contract["outer_folds"])))
    if verification.get("selection", {}).get("folds") != expected_folds:
        raise ValueError("Step 5 verification did not cover all folds")
    if tuple(verification.get("selection", {}).get("arms", ())) != tuple(ALL_ARMS):
        raise ValueError("Step 5 verification did not cover all seven arms")
    observed = verification.get("observed", {})
    expected_jobs = int(representation_contract["expected_extraction_jobs"])
    if observed.get("expected_jobs") != expected_jobs or observed.get("observed_jobs") != expected_jobs:
        raise ValueError("Step 5 verification job count does not match its contract")


def validate_probe_contract(contract: Mapping[str, Any]) -> None:
    validate_embedded_hash(contract, hash_field="contract_sha256", label="Step 6 probe contract")
    if contract.get("status") != "frozen_before_future_probe_training":
        raise ValueError("Step 6 probe contract has an invalid status")
    if tuple(contract.get("arms", ())) != tuple(ALL_ARMS):
        raise ValueError("Step 6 arm order changed")
    if tuple(contract.get("probe", {}).get("l2_grid", ())) != L2_GRID:
        raise ValueError("Step 6 L2 grid changed")
    gates = contract.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values()):
        raise ValueError("one or more Step 6 contract gates failed")
    for source in contract.get("sources", {}).values():
        path = Path(str(source.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(source.get("sha256", "")):
            raise ValueError(f"Step 6 source changed: {path}")
    for job in contract.get("jobs", []):
        report_path = Path(str(job.get("representation_run_path", ""))).expanduser().resolve()
        if sha256_file(report_path) != str(job.get("representation_run_sha256", "")):
            raise ValueError(f"Step 5 report changed: fold {job.get('fold')} {job.get('regime')}")
        for partition in PARTITIONS:
            artifact = job["artifacts"][partition]
            vector_path = Path(str(artifact["representations_path"])).expanduser().resolve()
            rows_path = Path(str(artifact["rows_path"])).expanduser().resolve()
            if sha256_file(vector_path) != str(artifact["representations_sha256"]):
                raise ValueError(f"Step 5 vector changed: {vector_path}")
            if sha256_file(rows_path) != str(artifact["rows_sha256"]):
                raise ValueError(f"Step 5 rows changed: {rows_path}")


def write_probe_contract(output_directory: Path, contract: Mapping[str, Any]) -> None:
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    (output / "probe-plan.md").write_text(render_probe_plan_markdown(contract), encoding="utf-8")


def render_probe_plan_markdown(contract: Mapping[str, Any]) -> str:
    probe = contract["probe"]
    estimand = contract["confirmatory_estimand"]
    lines = [
        "# Identical Future-Probe Plan",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Folds | {contract['outer_folds']} |",
        f"| Arms | {len(contract['arms'])} |",
        f"| Probe jobs | {contract['expected_probe_jobs']} |",
        f"| Architecture | `{probe['architecture']}` |",
        f"| L2 candidates | {len(probe['l2_grid'])} |",
        f"| Primary metric | `{contract['metrics']['primary']}` |",
        f"| Bootstrap replicates | {estimand['bootstrap_replicates']:,} |",
        "",
        "## Confirmatory estimand",
        "",
        f"`{estimand['primary_comparison']}`",
        "",
        "A positive value means the authentic-preference representation improves the pooled",
        "out-of-fold probabilistic forecast relative to the untouched generic encoder.",
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
