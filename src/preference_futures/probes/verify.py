"""Independent verification and aggregation for Step 6 future probes."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.probes.common import (
    L2_GRID,
    PROBE_VERIFICATION_SCHEMA_VERSION,
    parse_arm_selection,
    select_l2_candidate,
)
from preference_futures.probes.contract import validate_probe_contract
from preference_futures.probes.metrics import binary_metrics, per_record_log_losses
from preference_futures.selection.diagnostics import ALL_ARMS
from preference_futures.training.common import (
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
)

FLOAT_TOLERANCE = 1e-5


def verify_probe_runs(
    probe_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify persisted probes and build the full out-of-fold summary when complete."""

    torch, load_file = _require_verify_stack()
    root = probe_directory.expanduser().resolve()
    contract = load_json(root / "contract.json")
    errors: list[str] = []
    try:
        validate_probe_contract(contract)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"probe contract is invalid: {exc}")

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
        errors.append("selected Step 6 jobs do not match the frozen contract")

    labels = _load_future_labels(Path(contract["sources"]["episodes"]["path"]))
    observed_keys: set[tuple[int, str]] = set()
    environments: set[tuple[str, ...]] = set()
    device_types: set[str] = set()
    selected_lambdas: Counter[str] = Counter()
    run_summaries: list[dict[str, Any]] = []
    predictions_by_arm: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    baseline_by_fold: dict[int, dict[str, Any]] = {}

    for key in sorted(expected_keys):
        fold, regime = key
        job = jobs[key]
        run_directory = root / "runs" / f"fold-{fold:02d}" / regime
        report_path = run_directory / "run.json"
        if not report_path.exists():
            errors.append(f"missing Step 6 run report: fold {fold} {regime}")
            continue
        try:
            report = load_json(report_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid Step 6 run report: fold {fold} {regime}: {exc}")
            continue
        observed_keys.add(key)
        if report.get("status") != "complete":
            errors.append(f"incomplete Step 6 run: fold {fold} {regime}")
        if report.get("contract_sha256") != contract.get("contract_sha256"):
            errors.append(f"Step 6 contract mismatch: fold {fold} {regime}")
        if report.get("representation_run_sha256") != job.get("representation_run_sha256"):
            errors.append(f"Step 5 run mismatch: fold {fold} {regime}")

        artifacts = report.get("artifacts", {})
        probe_path = run_directory / "probe.safetensors"
        validation_predictions_path = run_directory / "validation.predictions.jsonl"
        test_predictions_path = run_directory / "test.predictions.jsonl"
        for path, hash_field in (
            (probe_path, "probe_sha256"),
            (validation_predictions_path, "validation_predictions_sha256"),
            (test_predictions_path, "test_predictions_sha256"),
        ):
            if not path.exists() or sha256_file(path) != artifacts.get(hash_field):
                errors.append(f"Step 6 artifact hash mismatch: fold {fold} {regime} {path.name}")

        candidates = report.get("probe", {}).get("candidates", [])
        candidate_lambdas = tuple(float(candidate.get("l2_lambda")) for candidate in candidates)
        if candidate_lambdas != L2_GRID:
            errors.append(f"L2 candidate grid mismatch: fold {fold} {regime}")
        try:
            expected_selected = select_l2_candidate(candidates)
            selected_lambda = float(report["probe"]["selected_l2_lambda"])
            if selected_lambda != float(expected_selected["l2_lambda"]):
                errors.append(f"validation selection mismatch: fold {fold} {regime}")
            selected_lambdas[repr(selected_lambda)] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid validation selection: fold {fold} {regime}: {exc}")
            continue

        tensors = load_file(str(probe_path), device="cpu")
        if set(tensors) != {"weight", "bias", "feature_mean", "feature_scale"}:
            errors.append(f"unexpected probe tensors: fold {fold} {regime}")
            continue
        weight = tensors["weight"].float()
        bias = tensors["bias"].float().reshape(())
        feature_mean = tensors["feature_mean"].float()
        feature_scale = tensors["feature_scale"].float()
        if (
            weight.ndim != 1
            or feature_mean.shape != weight.shape
            or feature_scale.shape != weight.shape
            or not bool(torch.isfinite(weight).all().item())
            or not bool(torch.isfinite(bias).item())
            or not bool(torch.isfinite(feature_mean).all().item())
            or not bool(torch.isfinite(feature_scale).all().item())
            or not bool((feature_scale > 0).all().item())
        ):
            errors.append(f"invalid probe tensors: fold {fold} {regime}")
            continue

        recomputed: dict[str, dict[str, Any]] = {}
        for partition, predictions_path in (
            ("validation", validation_predictions_path),
            ("test", test_predictions_path),
        ):
            source = job["artifacts"][partition]
            matrix_path = Path(str(source["representations_path"]))
            rows_path = Path(str(source["rows_path"]))
            matrix_tensors = load_file(str(matrix_path), device="cpu")
            matrix = matrix_tensors["representations"].float()
            rows = load_jsonl(rows_path)
            prediction_rows = load_jsonl(predictions_path)
            if len(rows) != len(prediction_rows) or int(matrix.shape[0]) != len(rows):
                errors.append(f"prediction row count mismatch: fold {fold} {regime} {partition}")
                continue
            expected_labels = []
            for index, (row, prediction) in enumerate(zip(rows, prediction_rows, strict=True)):
                episode_id = str(row.get("episode_id", ""))
                if (
                    int(row.get("row_index", -1)) != index
                    or int(prediction.get("row_index", -1)) != index
                    or str(prediction.get("episode_id", "")) != episode_id
                    or str(prediction.get("lineage_id", "")) != str(row.get("lineage_id", ""))
                ):
                    errors.append(f"prediction identity mismatch: fold {fold} {regime} {partition}")
                    break
                expected_label = labels.get(episode_id)
                if expected_label is None or bool(prediction.get("future_revised")) != bool(expected_label):
                    errors.append(f"prediction label mismatch: fold {fold} {regime} {partition}")
                    break
                expected_labels.append(expected_label)

            standardised = (matrix - feature_mean) / feature_scale
            logits = standardised @ weight + bias
            probabilities = logits.sigmoid()
            stored_logits = [float(row["logit"]) for row in prediction_rows]
            stored_probabilities = [float(row["probability"]) for row in prediction_rows]
            if not _sequence_close(logits.tolist(), stored_logits):
                errors.append(f"stored logits mismatch: fold {fold} {regime} {partition}")
            if not _sequence_close(probabilities.tolist(), stored_probabilities):
                errors.append(f"stored probabilities mismatch: fold {fold} {regime} {partition}")
            metrics = binary_metrics(expected_labels, probabilities.tolist())
            reported_metrics = report.get(partition, {})
            if not _metrics_close(metrics, reported_metrics):
                errors.append(f"reported metrics mismatch: fold {fold} {regime} {partition}")
            recomputed[partition] = metrics

            if partition == "test":
                for row, label, probability in zip(
                    rows,
                    expected_labels,
                    probabilities.tolist(),
                    strict=True,
                ):
                    episode_id = str(row["episode_id"])
                    if episode_id in predictions_by_arm[regime]:
                        errors.append(f"duplicate out-of-fold prediction: {regime} {episode_id}")
                    predictions_by_arm[regime][episode_id] = {
                        "episode_id": episode_id,
                        "lineage_id": str(row["lineage_id"]),
                        "future_revised": label,
                        "probability": float(probability),
                        "fold": fold,
                    }

        baseline = report.get("constant_prior_baseline", {}).get("test", {})
        if fold not in baseline_by_fold:
            baseline_by_fold[fold] = dict(baseline)
        elif not _metrics_close(baseline_by_fold[fold], baseline):
            errors.append(f"constant-prior baseline differs across arms: fold {fold}")

        environment = report.get("environment", {})
        device_type = str(environment.get("device", "")).split(":", maxsplit=1)[0]
        device_types.add(device_type)
        environments.add(
            (
                str(environment.get("python", "")),
                str(environment.get("torch", "")),
                str(environment.get("device_name", "")),
            )
        )
        run_summaries.append(
            {
                "fold": fold,
                "regime": regime,
                "selected_l2_lambda": selected_lambda,
                "validation": recomputed.get("validation"),
                "test": recomputed.get("test"),
            }
        )

    if len(device_types) != 1:
        errors.append(f"expected one Step 6 device type, observed {sorted(device_types)}")
    if len(environments) != 1:
        errors.append("expected one Step 6 runtime environment")

    full_selection = (
        selected_folds == tuple(range(outer_folds))
        and tuple(selected_arms) == tuple(ALL_ARMS)
    )
    summary = None
    if full_selection:
        summary_errors, summary = _build_full_summary(
            contract=contract,
            labels=labels,
            predictions_by_arm=predictions_by_arm,
            run_summaries=run_summaries,
            baseline_by_fold=baseline_by_fold,
        )
        errors.extend(summary_errors)

    checks = {
        "probe_contract_is_valid": not any(
            error.startswith("probe contract is invalid") for error in errors
        ),
        "all_expected_runs_exist": observed_keys == expected_keys,
        "all_runs_are_complete": not any(error.startswith("incomplete Step 6 run") for error in errors),
        "all_artifact_hashes_match": not any("artifact hash mismatch" in error for error in errors),
        "all_jobs_used_the_frozen_l2_grid": not any("L2 candidate grid mismatch" in error for error in errors),
        "validation_only_selection_is_reproducible": not any("selection mismatch" in error for error in errors),
        "saved_probe_predictions_recompute": not any(
            "stored logits mismatch" in error or "stored probabilities mismatch" in error
            for error in errors
        ),
        "reported_metrics_recompute": not any("reported metrics mismatch" in error for error in errors),
        "prediction_rows_and_labels_match": not any(
            "prediction identity mismatch" in error or "prediction label mismatch" in error
            for error in errors
        ),
        "constant_baseline_is_identical_across_arms": not any(
            "constant-prior baseline differs" in error for error in errors
        ),
        "one_device_type_used": len(device_types) == 1,
        "one_runtime_environment_used": len(environments) == 1,
        "full_out_of_fold_coverage": not any(
            "out-of-fold" in error or "coverage" in error for error in errors
        ),
    }
    report = {
        "probe_verification_schema_version": PROBE_VERIFICATION_SCHEMA_VERSION,
        "status": "pass" if not errors and all(checks.values()) else "fail",
        "passed": not errors and all(checks.values()),
        "contract_sha256": contract.get("contract_sha256"),
        "selection": {"folds": list(selected_folds), "arms": list(selected_arms)},
        "observed": {
            "expected_jobs": len(expected_keys),
            "observed_jobs": len(observed_keys),
            "device_types": sorted(device_types),
            "runtime_environment_count": len(environments),
            "selected_l2_counts": dict(sorted(selected_lambdas.items())),
        },
        "checks": checks,
        "errors": errors,
        "runs": run_summaries,
    }
    return report, summary


def _build_full_summary(
    *,
    contract: Mapping[str, Any],
    labels: Mapping[str, int],
    predictions_by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    run_summaries: Sequence[Mapping[str, Any]],
    baseline_by_fold: Mapping[int, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    expected_ids = set(labels)
    arm_metrics: dict[str, Any] = {}
    ordered_ids = sorted(expected_ids)
    for arm in ALL_ARMS:
        predictions = predictions_by_arm.get(arm, {})
        if set(predictions) != expected_ids:
            errors.append(f"out-of-fold coverage mismatch for arm {arm}")
            continue
        arm_labels = [int(predictions[episode_id]["future_revised"]) for episode_id in ordered_ids]
        if arm_labels != [labels[episode_id] for episode_id in ordered_ids]:
            errors.append(f"out-of-fold label mismatch for arm {arm}")
            continue
        probabilities = [float(predictions[episode_id]["probability"]) for episode_id in ordered_ids]
        metrics = binary_metrics(arm_labels, probabilities)
        fold_rows = [row for row in run_summaries if row["regime"] == arm]
        generic_rows = {
            int(row["fold"]): row for row in run_summaries if row["regime"] == "generic"
        }
        folds_beating_generic = None
        if arm != "generic" and len(fold_rows) == int(contract["outer_folds"]):
            folds_beating_generic = sum(
                float(generic_rows[int(row["fold"])]["test"]["log_loss"])
                > float(row["test"]["log_loss"])
                for row in fold_rows
            )
        arm_metrics[arm] = {
            **metrics,
            "fold_log_losses": [
                {
                    "fold": int(row["fold"]),
                    "log_loss": float(row["test"]["log_loss"]),
                }
                for row in sorted(fold_rows, key=lambda item: int(item["fold"]))
            ],
            "folds_beating_generic": folds_beating_generic,
        }

    if "generic" in arm_metrics:
        generic_loss = float(arm_metrics["generic"]["log_loss"])
        for arm, metrics in arm_metrics.items():
            metrics["log_loss_improvement_vs_generic"] = generic_loss - float(
                metrics["log_loss"]
            )

    comparisons = {}
    if all(arm in predictions_by_arm for arm in ALL_ARMS):
        for comparator in (
            "generic",
            "language_adaptation",
            "pair_exposure",
            "temporal_direction",
            "random_label",
            "shuffled_preference",
        ):
            comparisons[f"authentic_preference_vs_{comparator}"] = _lineage_bootstrap(
                authentic=predictions_by_arm["authentic_preference"],
                comparator=predictions_by_arm[comparator],
                seed=int(contract["confirmatory_estimand"]["bootstrap_seed"]),
                replicates=int(
                    contract["confirmatory_estimand"]["bootstrap_replicates"]
                ),
            )

    weighted_baseline_loss = 0.0
    weighted_baseline_brier = 0.0
    baseline_records = 0
    for baseline in baseline_by_fold.values():
        records = int(baseline.get("records", 0))
        weighted_baseline_loss += float(baseline.get("log_loss", 0.0)) * records
        weighted_baseline_brier += float(baseline.get("brier_score", 0.0)) * records
        baseline_records += records
    constant_baseline = {
        "records": baseline_records,
        "log_loss": weighted_baseline_loss / baseline_records if baseline_records else None,
        "brier_score": weighted_baseline_brier / baseline_records if baseline_records else None,
    }
    return errors, {
        "status": "complete" if not errors else "invalid",
        "contract_sha256": contract["contract_sha256"],
        "episodes": len(labels),
        "lineages": len(
            {
                str(row["lineage_id"])
                for row in predictions_by_arm.get("generic", {}).values()
            }
        ),
        "primary_metric": contract["metrics"]["primary"],
        "arm_metrics": arm_metrics,
        "constant_prior_baseline": constant_baseline,
        "paired_lineage_bootstrap": comparisons,
        "interpretation_guard": {
            "positive_improvement_means_lower_log_loss": True,
            "primary_comparison": "authentic_preference versus generic",
            "step_6_tests_linear_decodability": True,
            "step_6_does_not_establish_general_nonlinear_impossibility": True,
        },
    }


def _lineage_bootstrap(
    *,
    authentic: Mapping[str, Mapping[str, Any]],
    comparator: Mapping[str, Mapping[str, Any]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    episode_ids = sorted(authentic)
    if set(comparator) != set(episode_ids):
        raise ValueError("paired bootstrap requires identical episode IDs")
    labels = [int(authentic[episode_id]["future_revised"]) for episode_id in episode_ids]
    authentic_probabilities = [float(authentic[episode_id]["probability"]) for episode_id in episode_ids]
    comparator_probabilities = [float(comparator[episode_id]["probability"]) for episode_id in episode_ids]
    authentic_losses = per_record_log_losses(labels, authentic_probabilities)
    comparator_losses = per_record_log_losses(labels, comparator_probabilities)

    lineage_sums: dict[str, float] = defaultdict(float)
    lineage_counts: Counter[str] = Counter()
    for episode_id, authentic_loss, comparator_loss in zip(
        episode_ids,
        authentic_losses,
        comparator_losses,
        strict=True,
    ):
        lineage = str(authentic[episode_id]["lineage_id"])
        if lineage != str(comparator[episode_id]["lineage_id"]):
            raise ValueError("paired bootstrap lineage mismatch")
        lineage_sums[lineage] += comparator_loss - authentic_loss
        lineage_counts[lineage] += 1
    lineages = sorted(lineage_sums)
    point = sum(lineage_sums.values()) / sum(lineage_counts.values())
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        sampled = rng.choices(lineages, k=len(lineages))
        numerator = sum(lineage_sums[lineage] for lineage in sampled)
        denominator = sum(lineage_counts[lineage] for lineage in sampled)
        samples.append(numerator / denominator)
    samples.sort()
    lower = _percentile(samples, 0.025)
    upper = _percentile(samples, 0.975)
    return {
        "log_loss_improvement": point,
        "confidence_interval_95": [lower, upper],
        "bootstrap_replicates": replicates,
        "lineages": len(lineages),
        "probability_improvement_positive": sum(sample > 0.0 for sample in samples)
        / replicates,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def _metrics_close(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in observed.items():
        other = expected.get(key)
        if value is None or other is None:
            if value is not None or other is not None:
                return False
        elif isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if not math.isclose(float(value), float(other), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
                return False
        elif value != other:
            return False
    return True


def _sequence_close(first: Sequence[float], second: Sequence[float]) -> bool:
    return len(first) == len(second) and all(
        math.isclose(float(left), float(right), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
        for left, right in zip(first, second, strict=True)
    )


def _load_future_labels(path: Path) -> dict[str, int]:
    labels = {}
    for record in load_jsonl(path):
        episode_id = str(record.get("episode_id", ""))
        target = record.get("future_revised")
        if not episode_id or episode_id in labels or type(target) is not bool:
            raise ValueError("invalid Step 6 future-label source")
        labels[episode_id] = int(target)
    return labels


def write_probe_verification(
    probe_directory: Path,
    report: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
) -> None:
    root = probe_directory.expanduser().resolve()
    write_json(root / "probe-verification.json", report)
    (root / "probe-verification.md").write_text(
        render_probe_verification_markdown(report),
        encoding="utf-8",
    )
    if summary is not None:
        write_json(root / "probe-summary.json", summary)
        (root / "probe-summary.md").write_text(
            render_probe_summary_markdown(summary),
            encoding="utf-8",
        )


def render_probe_verification_markdown(report: Mapping[str, Any]) -> str:
    observed = report["observed"]
    lines = [
        "# Identical Future-Probe Verification",
        "",
        f"**Status:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Expected jobs | {observed['expected_jobs']} |",
        f"| Observed jobs | {observed['observed_jobs']} |",
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
    lines.extend(f"- {error}" for error in report["errors"])
    if not report["errors"]:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def render_probe_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Step 6 Future-Probe Summary",
        "",
        "| Arm | Log loss | Improvement vs generic | Brier | ROC AUC | Folds beating generic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ALL_ARMS:
        row = summary["arm_metrics"].get(arm, {})
        auc = row.get("roc_auc")
        lines.append(
            f"| {arm} | {float(row.get('log_loss', float('nan'))):.6f} | "
            f"{float(row.get('log_loss_improvement_vs_generic', float('nan'))):+.6f} | "
            f"{float(row.get('brier_score', float('nan'))):.6f} | "
            f"{float(auc):.6f} | {row.get('folds_beating_generic', '—')} |"
            if auc is not None
            else f"| {arm} | — | — | — | — | — |"
        )
    lines.extend(["", "## Paired lineage bootstrap", ""])
    for name, comparison in summary["paired_lineage_bootstrap"].items():
        lower, upper = comparison["confidence_interval_95"]
        lines.append(
            f"- `{name}`: improvement {comparison['log_loss_improvement']:+.6f}; "
            f"95% CI [{lower:+.6f}, {upper:+.6f}]."
        )
    lines.extend(
        [
            "",
            "Positive improvement means lower log loss for the authentic-preference probe.",
            "",
        ]
    )
    return "\n".join(lines)


def _require_verify_stack() -> tuple[Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Step 6 verification dependencies are missing. Install with: "
            "python -m pip install -e '.[train]'"
        ) from exc
    return torch, load_file
