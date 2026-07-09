"""Step 4 source-task diagnostics and frozen encoder selection."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.training.common import (
    LANGUAGE_ADAPTATION_REGIME,
    TRAINED_REGIMES,
    canonical_json_sha256,
    load_json,
    load_jsonl,
    sha256_directory,
    sha256_file,
    write_json,
)
from preference_futures.training.contract import validate_training_contract

SELECTION_SCHEMA_VERSION = 1
SOURCE_TASK_SUMMARY_SCHEMA_VERSION = 1
TRAJECTORY_SUMMARY_SCHEMA_VERSION = 1
ENCODER_HASH_AUDIT_SCHEMA_VERSION = 1
GENERIC_REGIME = "generic"
ALL_ARMS = (GENERIC_REGIME, *TRAINED_REGIMES)
WILSON_Z_95 = 1.959963984540054


def freeze_encoder_selection(
    training_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Diagnose all Step 3 runs and freeze the seven-arm Step 5 encoder manifest."""

    training = training_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    contract = load_json(training / "contract.json")
    validate_training_contract(contract)
    persisted_verification_path = training / "training-verification-confirmatory.json"
    persisted_verification = load_json(persisted_verification_path)
    _require_complete_verification(persisted_verification, contract)

    jobs = {
        (int(job["fold"]), str(job["regime"])): job for job in contract["jobs"]
    }
    entries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    diagnostic_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trained_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for fold in range(int(contract["outer_folds"])):
        for regime in TRAINED_REGIMES:
            job = jobs[(fold, regime)]
            run_directory = training / "runs" / f"fold-{fold:02d}" / regime
            report = load_json(run_directory / "run.json")
            metrics = load_jsonl(run_directory / "metrics.jsonl")
            _validate_run_identity(report, contract, fold=fold, regime=regime)
            trajectory = _summarise_trajectory(
                metrics,
                expected_final_step=int(contract["optimisation"]["update_steps"]),
                fold=fold,
                regime=regime,
            )
            trajectories.append(trajectory)

            encoder_path = run_directory / "encoder"
            encoder_sha256 = sha256_directory(encoder_path)
            artifact_valid = encoder_sha256 == str(
                report.get("artifacts", {}).get("encoder_sha256", "")
            )
            if not artifact_valid:
                raise ValueError(f"encoder hash changed: fold {fold} {regime}")

            if regime == LANGUAGE_ADAPTATION_REGIME:
                diagnostic = _language_diagnostic(report, trajectory)
            else:
                validation_records = load_jsonl(Path(str(job["validation"]["path"])))
                diagnostic = _classification_diagnostic(report, validation_records)
            diagnostic_rows[regime].append(diagnostic)

            flags = list(diagnostic["diagnostic_flags"])
            if not diagnostic["source_task_learned"]:
                flags.append("source_task_not_demonstrably_learned")
            entry = {
                "fold": fold,
                "regime": regime,
                "arm_kind": "trained",
                "encoder_path": str(encoder_path),
                "encoder_sha256": encoder_sha256,
                "artifact_valid": artifact_valid,
                "source_task_status": diagnostic["source_task_status"],
                "source_task_learned": diagnostic["source_task_learned"],
                "eligible_for_downstream": artifact_valid,
                "diagnostic_flags": sorted(set(flags)),
                "validation": report["validation"],
            }
            entries.append(entry)
            trained_hashes[encoder_sha256].append({"fold": fold, "regime": regime})

    base_encoder_path = Path(str(contract["model"]["base_snapshot_path"])) / "encoder"
    base_encoder_sha256 = sha256_directory(base_encoder_path)
    for fold in range(int(contract["outer_folds"])):
        entries.append(
            {
                "fold": fold,
                "regime": GENERIC_REGIME,
                "arm_kind": "untouched_base",
                "encoder_path": str(base_encoder_path),
                "encoder_sha256": base_encoder_sha256,
                "artifact_valid": True,
                "source_task_status": "not_trained",
                "source_task_learned": None,
                "eligible_for_downstream": True,
                "diagnostic_flags": ["shared_frozen_base_snapshot"],
                "validation": None,
            }
        )

    hash_audit = _build_hash_audit(trained_hashes, base_encoder_sha256)
    if not hash_audit["passed"]:
        raise ValueError(f"encoder hash audit failed: {hash_audit['errors']}")

    aggregate = {
        regime: _aggregate_regime(regime, rows)
        for regime, rows in diagnostic_rows.items()
    }
    source_task_summary = {
        "source_task_summary_schema_version": SOURCE_TASK_SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "persisted_verification_sha256": sha256_file(persisted_verification_path),
        "folds": int(contract["outer_folds"]),
        "trained_regimes": list(TRAINED_REGIMES),
        "aggregate": aggregate,
        "interpretation": {
            "source_task_learning_is_diagnostic": True,
            "source_task_failure_does_not_remove_a_preregistered_control_arm": True,
            "future_transfer_has_not_been_tested": True,
        },
    }
    trajectory_summary = {
        "trajectory_summary_schema_version": TRAJECTORY_SUMMARY_SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "runs": trajectories,
        "aggregate": _aggregate_trajectories(trajectories),
    }
    manifest = {
        "encoder_selection_schema_version": SELECTION_SCHEMA_VERSION,
        "status": "frozen_for_step_5",
        "contract_sha256": contract["contract_sha256"],
        "source_task_summary_sha256": canonical_json_sha256(source_task_summary),
        "selection_policy": {
            "rule": (
                "Downstream eligibility is determined by mechanical artifact validity, not by "
                "source-task success. Source-task results remain diagnostic labels."
            ),
            "exclude_only_if": [
                "missing_or_changed_encoder_artifact",
                "invalid_or_incomplete_confirmatory_run",
                "non_finite_validation_metrics",
            ],
            "checkpoint_selection": "fixed final update 600 only",
            "post_result_retraining": False,
        },
        "counts": {
            "folds": int(contract["outer_folds"]),
            "arms_per_fold": len(ALL_ARMS),
            "entries": len(entries),
            "eligible_entries": sum(bool(item["eligible_for_downstream"]) for item in entries),
        },
        "arms": list(ALL_ARMS),
        "entries": sorted(entries, key=lambda item: (int(item["fold"]), str(item["regime"]))),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "source-task-summary.json", source_task_summary)
    (output / "source-task-summary.md").write_text(
        render_source_task_summary_markdown(source_task_summary), encoding="utf-8"
    )
    write_json(output / "accepted-encoders.json", manifest)
    write_json(output / "encoder-hash-audit.json", hash_audit)
    write_json(output / "trajectory-summary.json", trajectory_summary)
    return manifest


def _require_complete_verification(
    verification: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    expected_folds = list(range(int(contract["outer_folds"])))
    expected_regimes = list(TRAINED_REGIMES)
    if verification.get("passed") is not True or verification.get("mode") != "confirmatory":
        raise ValueError("Step 3 confirmatory verification has not passed")
    selection = verification.get("selection", {})
    if selection.get("folds") != expected_folds or selection.get("regimes") != expected_regimes:
        raise ValueError("Step 3 verification did not cover every fold and regime")
    observed = verification.get("observed", {})
    expected_jobs = int(contract["expected_training_jobs"])
    if (
        observed.get("expected_jobs") != expected_jobs
        or observed.get("observed_jobs") != expected_jobs
    ):
        raise ValueError("Step 3 verification job count does not match the contract")


def _validate_run_identity(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    fold: int,
    regime: str,
) -> None:
    if report.get("status") != "complete" or report.get("non_confirmatory") is not False:
        raise ValueError(f"invalid confirmatory run status: fold {fold} {regime}")
    if report.get("fold") != fold or report.get("regime") != regime:
        raise ValueError(f"run identity mismatch: fold {fold} {regime}")
    if report.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError(f"run contract mismatch: fold {fold} {regime}")
    validation = report.get("validation", {})
    for name in ("mean_loss", "accuracy"):
        value = validation.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite validation {name}: fold {fold} {regime}")


def _summarise_trajectory(
    metrics: Sequence[Mapping[str, Any]],
    *,
    expected_final_step: int,
    fold: int,
    regime: str,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError(f"missing training trajectory: fold {fold} {regime}")
    steps = [int(item["step"]) for item in metrics]
    if steps != sorted(set(steps)) or steps[-1] != expected_final_step:
        raise ValueError(f"invalid training trajectory steps: fold {fold} {regime}")
    losses = [float(item["mean_training_loss_since_last_log"]) for item in metrics]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError(f"non-finite training trajectory: fold {fold} {regime}")
    first = losses[0]
    final = losses[-1]
    return {
        "fold": fold,
        "regime": regime,
        "logged_windows": len(metrics),
        "first_step": steps[0],
        "final_step": steps[-1],
        "first_logged_loss": first,
        "final_logged_loss": final,
        "minimum_logged_loss": min(losses),
        "maximum_logged_loss": max(losses),
        "absolute_loss_reduction": first - final,
        "relative_loss_reduction": (first - final) / first if first else None,
        "loss_decreased": final < first,
    }


def _classification_diagnostic(
    report: Mapping[str, Any], validation_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    targets = [int(record["target"]) for record in validation_records]
    if not targets or any(target not in (0, 1) for target in targets):
        raise ValueError("classification validation targets are missing or invalid")
    counts = Counter(targets)
    records = len(targets)
    accuracy = float(report["validation"]["accuracy"])
    correct = int(round(accuracy * records))
    observed_accuracy = correct / records
    if not math.isclose(observed_accuracy, accuracy, abs_tol=1e-12):
        raise ValueError("validation accuracy does not resolve to an integer correct count")
    positive_rate = counts[1] / records
    prior_accuracy = max(positive_rate, 1.0 - positive_rate)
    prior_log_loss = _binary_entropy(positive_rate)
    lower, upper = _wilson_interval(correct, records)
    mean_loss = float(report["validation"]["mean_loss"])
    if lower > prior_accuracy and mean_loss < prior_log_loss:
        status = "learned_above_prior"
    elif upper < prior_accuracy:
        status = "below_prior"
    else:
        status = "null_like"
    flags = []
    if lower <= prior_accuracy <= upper:
        flags.append("accuracy_interval_contains_class_prior")
    elif lower > prior_accuracy:
        flags.append("accuracy_interval_above_class_prior")
    else:
        flags.append("accuracy_interval_below_class_prior")
    flags.append(
        "validation_loss_below_class_prior"
        if mean_loss < prior_log_loss
        else "validation_loss_not_below_class_prior"
    )
    return {
        "records": records,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_interval_95": [lower, upper],
        "target_counts": {"0": counts[0], "1": counts[1]},
        "positive_rate": positive_rate,
        "class_prior_accuracy": prior_accuracy,
        "class_prior_log_loss": prior_log_loss,
        "mean_loss": mean_loss,
        "source_task_status": status,
        "source_task_learned": status == "learned_above_prior",
        "diagnostic_flags": flags,
    }


def _language_diagnostic(
    report: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    validation = report["validation"]
    fallback = int(
        validation.get("mask_fallback_examples", report.get("mask_fallback_examples", 0))
    )
    learned = bool(trajectory["loss_decreased"]) and fallback == 0
    flags = [
        "training_loss_decreased"
        if trajectory["loss_decreased"]
        else "training_loss_not_decreased"
    ]
    flags.append("zero_mask_fallbacks" if fallback == 0 else "mask_fallbacks_present")
    return {
        "records": int(validation["records"]),
        "supervised_units": int(validation["supervised_units"]),
        "accuracy": float(validation["accuracy"]),
        "mean_loss": float(validation["mean_loss"]),
        "perplexity": float(validation["perplexity"]),
        "mask_fallback_examples": fallback,
        "source_task_status": "learned" if learned else "diagnostic_failure",
        "source_task_learned": learned,
        "diagnostic_flags": flags,
    }


def _aggregate_regime(regime: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if regime == LANGUAGE_ADAPTATION_REGIME:
        units = sum(int(row["supervised_units"]) for row in rows)
        mean_loss = (
            sum(float(row["mean_loss"]) * int(row["supervised_units"]) for row in rows)
            / units
        )
        accuracy = (
            sum(float(row["accuracy"]) * int(row["supervised_units"]) for row in rows)
            / units
        )
        learned_folds = sum(bool(row["source_task_learned"]) for row in rows)
        return {
            "folds": len(rows),
            "supervised_units": units,
            "accuracy": accuracy,
            "mean_loss": mean_loss,
            "perplexity": math.exp(mean_loss),
            "mask_fallback_examples": sum(int(row["mask_fallback_examples"]) for row in rows),
            "source_task_status": "learned" if learned_folds == len(rows) else "mixed",
            "source_task_learned_folds": learned_folds,
        }

    records = sum(int(row["records"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    target_zero = sum(int(row["target_counts"]["0"]) for row in rows)
    target_one = sum(int(row["target_counts"]["1"]) for row in rows)
    positive_rate = target_one / records
    prior_accuracy = max(positive_rate, 1.0 - positive_rate)
    prior_log_loss = _binary_entropy(positive_rate)
    accuracy = correct / records
    lower, upper = _wilson_interval(correct, records)
    mean_loss = sum(float(row["mean_loss"]) * int(row["records"]) for row in rows) / records
    if lower > prior_accuracy and mean_loss < prior_log_loss:
        status = "learned_above_prior"
    elif upper < prior_accuracy:
        status = "below_prior"
    else:
        status = "null_like"
    return {
        "folds": len(rows),
        "records": records,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_interval_95": [lower, upper],
        "fold_accuracy_range": [
            min(float(row["accuracy"]) for row in rows),
            max(float(row["accuracy"]) for row in rows),
        ],
        "target_counts": {"0": target_zero, "1": target_one},
        "class_prior_accuracy": prior_accuracy,
        "class_prior_log_loss": prior_log_loss,
        "mean_loss": mean_loss,
        "source_task_status": status,
        "source_task_learned": status == "learned_above_prior",
        "fold_status_counts": dict(Counter(str(row["source_task_status"]) for row in rows)),
    }


def _aggregate_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory["regime"])].append(trajectory)
    return {
        regime: {
            "folds": len(rows),
            "mean_first_logged_loss": sum(float(row["first_logged_loss"]) for row in rows)
            / len(rows),
            "mean_final_logged_loss": sum(float(row["final_logged_loss"]) for row in rows)
            / len(rows),
            "folds_with_loss_decrease": sum(bool(row["loss_decreased"]) for row in rows),
        }
        for regime, rows in grouped.items()
    }


def _build_hash_audit(
    trained_hashes: Mapping[str, Sequence[Mapping[str, Any]]], base_encoder_sha256: str
) -> dict[str, Any]:
    duplicates = {
        digest: list(entries) for digest, entries in trained_hashes.items() if len(entries) > 1
    }
    base_collisions = list(trained_hashes.get(base_encoder_sha256, ()))
    errors = []
    if duplicates:
        errors.append("one or more trained encoders share the same hash")
    if base_collisions:
        errors.append("one or more trained encoders equal the untouched base encoder")
    return {
        "encoder_hash_audit_schema_version": ENCODER_HASH_AUDIT_SCHEMA_VERSION,
        "passed": not errors,
        "trained_encoder_count": sum(len(entries) for entries in trained_hashes.values()),
        "unique_trained_encoder_hashes": len(trained_hashes),
        "base_encoder_sha256": base_encoder_sha256,
        "duplicate_trained_hashes": duplicates,
        "trained_equal_to_base": base_collisions,
        "errors": errors,
    }


def _binary_entropy(positive_rate: float) -> float:
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return 0.0
    return -positive_rate * math.log(positive_rate) - (1.0 - positive_rate) * math.log(
        1.0 - positive_rate
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    z2 = WILSON_Z_95**2
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return centre - margin, centre + margin


def render_source_task_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Step 4 Source-Task Diagnostics",
        "",
        "**Status:** complete",
        "",
        "| Regime | Accuracy | Mean loss | Source-task status |",
        "|---|---:|---:|---|",
    ]
    for regime in TRAINED_REGIMES:
        row = summary["aggregate"][regime]
        lines.append(
            f"| {regime} | {float(row['accuracy']):.4%} | "
            f"{float(row['mean_loss']):.6f} | {row['source_task_status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            (
                "Source-task behaviour is diagnostic. A mechanically valid preregistered arm is "
                "retained for downstream representation extraction even when its source head is "
                "null-like. Future transfer has not yet been tested."
            ),
            "",
        ]
    )
    return "\n".join(lines)
