"""Independently verify persisted Step 3 training runs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.training.common import (
    TRAINING_VERIFICATION_SCHEMA_VERSION,
    load_json,
    parse_int_selection,
    parse_regime_selection,
    sha256_directory,
    write_json,
)
from preference_futures.training.contract import validate_training_contract


def verify_training_runs(
    training_directory: Path,
    *,
    folds: str = "all",
    regimes: str = "all",
    smoke: bool = False,
) -> dict[str, Any]:
    training = training_directory.expanduser().resolve()
    contract = load_json(training / "contract.json")
    errors: list[str] = []
    checks: dict[str, bool] = {}
    try:
        validate_training_contract(contract)
        checks["training_contract_is_valid"] = True
    except ValueError as exc:
        checks["training_contract_is_valid"] = False
        errors.append(str(exc))

    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_regimes = parse_regime_selection(regimes)
    root = training / ("smoke-runs" if smoke else "runs")
    expected_steps = None if smoke else int(contract["optimisation"]["update_steps"])
    expected_jobs = len(selected_folds) * len(selected_regimes)
    observed_jobs = 0
    reports: list[dict[str, Any]] = []
    artifact_hashes_match = True
    contract_hashes_match = True
    step_budgets_match = True
    padded_budgets_match = True
    checkpoint_rules_match = True
    no_early_stopping = True
    source_hashes_match = True
    initial_snapshots_match = True
    metrics_are_finite = True
    device_types: set[str] = set()
    fold_budgets: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)

    for fold in selected_folds:
        for regime in selected_regimes:
            directory = root / f"fold-{fold:02d}" / regime
            report_path = directory / "run.json"
            if not report_path.exists():
                errors.append(f"missing training run: {report_path}")
                continue
            observed_jobs += 1
            try:
                report = load_json(report_path)
            except (OSError, ValueError) as exc:
                errors.append(f"invalid run report {report_path}: {exc}")
                continue
            reports.append(report)
            if report.get("fold") != fold or report.get("regime") != regime:
                errors.append(f"run identity disagrees with path: {report_path}")
            contract_hashes_match = contract_hashes_match and (
                report.get("contract_sha256") == contract.get("contract_sha256")
            )
            optimisation = report.get("optimisation", {})
            steps = optimisation.get("optimizer_steps_completed")
            padded = optimisation.get("padded_token_positions")
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
                step_budgets_match = False
            if expected_steps is not None and steps != expected_steps:
                step_budgets_match = False
            expected_padded = (
                int(steps)
                * int(contract["optimisation"]["batch_size"])
                * int(contract["optimisation"]["maximum_sequence_length"])
                if isinstance(steps, int)
                else None
            )
            if padded != expected_padded:
                padded_budgets_match = False
            if isinstance(steps, int) and isinstance(padded, int):
                fold_budgets[fold][regime] = (steps, padded)
            checkpoint_rules_match = checkpoint_rules_match and (
                optimisation.get("checkpoint_step") == steps
            )
            no_early_stopping = no_early_stopping and (
                optimisation.get("early_stopping_used") is False
            )
            initial_snapshots_match = initial_snapshots_match and (
                report.get("model", {}).get("initial_encoder_snapshot_sha256")
                == contract.get("model", {}).get("base_snapshot_sha256")
            )
            environment = report.get("environment", {})
            device_value = str(environment.get("device", ""))
            if device_value:
                device_types.add(device_value.split(":", maxsplit=1)[0])

            job = _find_job(contract, fold, regime)
            source_files = report.get("source_files", {})
            source_hashes_match = source_hashes_match and (
                source_files.get("train_sha256") == job["train"]["sha256"]
                and source_files.get("validation_sha256")
                == job["validation"]["sha256"]
            )
            validation = report.get("validation", {})
            for key in ("mean_loss", "accuracy"):
                value = validation.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    metrics_are_finite = False

            artifacts = report.get("artifacts", {})
            try:
                current_hashes = {
                    "task_model_sha256": sha256_directory(directory / "task-model"),
                    "encoder_sha256": sha256_directory(directory / "encoder"),
                    "tokenizer_sha256": sha256_directory(directory / "tokenizer"),
                }
                artifact_hashes_match = artifact_hashes_match and current_hashes == artifacts
            except (OSError, ValueError):
                artifact_hashes_match = False

    checks["all_expected_runs_exist"] = observed_jobs == expected_jobs
    checks["run_contract_hashes_match"] = contract_hashes_match
    checks["source_corpus_hashes_match"] = source_hashes_match
    checks["initial_encoder_snapshot_matches"] = initial_snapshots_match
    checks["optimizer_update_budgets_match"] = step_budgets_match
    checks["padded_token_budgets_match"] = padded_budgets_match
    checks["fixed_final_checkpoint_rule_used"] = checkpoint_rules_match
    checks["no_task_specific_early_stopping"] = no_early_stopping
    checks["persisted_artifact_hashes_match"] = artifact_hashes_match
    checks["validation_metrics_are_finite"] = metrics_are_finite
    checks["one_device_type_used"] = len(device_types) <= 1
    checks["within_fold_regime_budgets_are_equal"] = _fold_budgets_equal(
        fold_budgets,
        selected_folds=selected_folds,
        selected_regimes=selected_regimes,
    )

    for name, passed in checks.items():
        if not passed:
            errors.append(name.replace("_", " "))
    passed = bool(checks) and all(checks.values()) and not errors
    report = {
        "training_verification_schema_version": TRAINING_VERIFICATION_SCHEMA_VERSION,
        "passed": passed,
        "mode": "smoke" if smoke else "confirmatory",
        "contract_sha256": contract.get("contract_sha256"),
        "selection": {
            "folds": list(selected_folds),
            "regimes": list(selected_regimes),
        },
        "observed": {
            "expected_jobs": expected_jobs,
            "observed_jobs": observed_jobs,
            "device_types": sorted(device_types),
            "run_reports_read": len(reports),
        },
        "checks": checks,
        "errors": sorted(set(errors)),
        "runs": [
            {
                "fold": item.get("fold"),
                "regime": item.get("regime"),
                "steps": item.get("optimisation", {}).get("optimizer_steps_completed"),
                "padded_token_positions": item.get("optimisation", {}).get(
                    "padded_token_positions"
                ),
                "validation": item.get("validation"),
                "encoder_sha256": item.get("artifacts", {}).get("encoder_sha256"),
            }
            for item in reports
        ],
    }
    return report


def write_training_verification(training_directory: Path, report: Mapping[str, Any]) -> None:
    training = training_directory.expanduser().resolve()
    suffix = "smoke" if report.get("mode") == "smoke" else "confirmatory"
    write_json(training / f"training-verification-{suffix}.json", report)
    (training / f"training-verification-{suffix}.md").write_text(
        render_training_verification_markdown(report), encoding="utf-8"
    )


def render_training_verification_markdown(report: Mapping[str, Any]) -> str:
    observed = report["observed"]
    lines = [
        "# Fixed-Budget Training Verification",
        "",
        f"**Mode:** {report['mode']}",
        f"**Status:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Observed",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Expected jobs | {observed['expected_jobs']:,} |",
        f"| Observed jobs | {observed['observed_jobs']:,} |",
        f"| Run reports read | {observed['run_reports_read']:,} |",
        f"| Device types | {', '.join(observed['device_types']) or 'none'} |",
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


def _find_job(contract: Mapping[str, Any], fold: int, regime: str) -> Mapping[str, Any]:
    for job in contract["jobs"]:
        if job.get("fold") == fold and job.get("regime") == regime:
            return job
    raise ValueError(f"contract has no job for fold {fold} regime {regime}")


def _fold_budgets_equal(
    fold_budgets: Mapping[int, Mapping[str, tuple[int, int]]],
    *,
    selected_folds: tuple[int, ...],
    selected_regimes: tuple[str, ...],
) -> bool:
    for fold in selected_folds:
        values = fold_budgets.get(fold, {})
        if set(values) != set(selected_regimes):
            return False
        if len(set(values.values())) != 1:
            return False
    return True
