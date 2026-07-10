"""Aggregate the final authentic-versus-shuffled MR.Q specificity check."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq.shuffled_common import (
    AUTHENTIC_TO_SHUFFLED,
    REQUIRED_NEGATIVE_REPLICATES,
    SHUFFLED_ARMS,
    comparison_passed,
    load_canonical_report,
    load_contract,
)
from preference_futures.editorial_mrq.transfer import (
    lineage_bootstrap_interval,
    paired_transfer_comparison,
)
from preference_futures.probes.metrics import binary_metrics, per_record_log_losses
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)


def aggregate_shuffled_control(
    control_directory: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Compare authentic MR.Q future losses with five shuffled-label controls."""

    root = control_directory.expanduser().resolve()
    contract = load_contract(root)
    output_json = root / "aggregate.json"
    output_markdown = root / "aggregate.md"
    if (output_json.exists() or output_markdown.exists()) and not force:
        raise ValueError(f"Step 8.7 aggregate exists; pass --force: {output_json}")

    transfer_root = Path(str(contract["sources"]["transfer_contract"]["path"])).parent
    outer_folds = int(contract["outer_folds"])
    authentic_predictions = {
        arm: _pool_predictions(transfer_root, arm, outer_folds=outer_folds)
        for arm in AUTHENTIC_TO_SHUFFLED
    }
    shuffled_predictions: dict[str, list[dict[str, tuple[int, float, str]]]] = {
        arm: [] for arm in SHUFFLED_ARMS
    }
    shuffled_metrics: dict[str, list[dict[str, Any]]] = {arm: [] for arm in SHUFFLED_ARMS}
    for replicate in range(int(contract["shuffle_replicates"])):
        for arm in SHUFFLED_ARMS:
            predictions = _pool_predictions(
                root,
                arm,
                outer_folds=outer_folds,
                replicate=replicate,
            )
            shuffled_predictions[arm].append(predictions)
            shuffled_metrics[arm].append(_metrics(predictions))

    authentic_metrics = {
        arm: _metrics(predictions) for arm, predictions in authentic_predictions.items()
    }
    bootstrap_seed = int(contract["estimand"]["bootstrap_seed"])
    bootstrap_replicates = int(contract["estimand"]["bootstrap_replicates"])
    individual: dict[str, list[dict[str, Any]]] = {}
    mean_comparisons: dict[str, dict[str, Any]] = {}
    negative_counts: dict[str, int] = {}
    for offset, (authentic_arm, shuffled_arm) in enumerate(AUTHENTIC_TO_SHUFFLED.items()):
        comparisons = []
        for replicate, control in enumerate(shuffled_predictions[shuffled_arm]):
            comparisons.append(
                paired_transfer_comparison(
                    authentic_predictions[authentic_arm],
                    control,
                    name=f"{authentic_arm}_minus_{shuffled_arm}_replicate_{replicate:02d}",
                    seed=bootstrap_seed + offset * 100 + replicate,
                    replicates=bootstrap_replicates,
                )
            )
        individual[authentic_arm] = comparisons
        negative_counts[authentic_arm] = sum(
            float(comparison["mean_log_loss_difference"]) < 0.0
            for comparison in comparisons
        )
        mean_comparisons[authentic_arm] = compare_authentic_to_mean_shuffled(
            authentic_predictions[authentic_arm],
            shuffled_predictions[shuffled_arm],
            name=f"{authentic_arm}_minus_mean_{shuffled_arm}",
            seed=bootstrap_seed + 1000 + offset,
            replicates=bootstrap_replicates,
        )

    primary_arm = "mrq_choice_aware"
    secondary_arm = "mrq_blind"
    primary_passed = comparison_passed(mean_comparisons[primary_arm])
    secondary_passed = comparison_passed(mean_comparisons[secondary_arm])
    primary_robust = negative_counts[primary_arm] >= REQUIRED_NEGATIVE_REPLICATES
    secondary_robust = negative_counts[secondary_arm] >= REQUIRED_NEGATIVE_REPLICATES
    supported = primary_passed and secondary_passed and primary_robust and secondary_robust

    matched_aggregate = load_canonical_report(
        Path(str(contract["sources"]["matched_aggregate"]["path"]))
    )
    report: dict[str, Any] = {
        "step_8_shuffled_control_aggregate_schema_version": 1,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "episodes": len(authentic_predictions[primary_arm]),
        "shuffle_replicates": int(contract["shuffle_replicates"]),
        "authentic_arms": authentic_metrics,
        "shuffled_arms": {
            arm: {
                "replicates": metrics,
                "mean_log_loss": statistics.fmean(
                    float(value["log_loss"]) for value in metrics
                ),
                "mean_accuracy": statistics.fmean(
                    float(value["accuracy"]) for value in metrics
                ),
            }
            for arm, metrics in shuffled_metrics.items()
        },
        "comparisons": {
            "mean": mean_comparisons,
            "individual_replicates": individual,
            "negative_point_estimate_counts": negative_counts,
        },
        "source_controls": _source_summaries(root, contract),
        "authentic_preference_specificity": {
            "supported": supported,
            "primary_mean_passed": primary_passed,
            "secondary_mean_passed": secondary_passed,
            "primary_negative_replicates": negative_counts[primary_arm],
            "secondary_negative_replicates": negative_counts[secondary_arm],
            "required_negative_replicates": REQUIRED_NEGATIVE_REPLICATES,
            "claim": (
                "Authentic preference supervision improves the compact MR.Q transfer state "
                "relative to shuffled-label MR.Q controls"
                if supported
                else "Authentic preference supervision did not establish compact-state transfer specificity"
            ),
        },
        "programme_context": {
            "matched_generic_specificity_supported": bool(
                matched_aggregate["compression_and_regularisation_specificity"]["supported"]
            ),
            "extended_generic_choice_aware_log_loss": float(
                matched_aggregate["arms"]["extended_generic_choice_aware"]["pooled_test"][
                    "log_loss"
                ]
            ),
            "note": (
                "Passing Step 8.7 would show that authentic labels shape the compact MR.Q state; "
                "it would not reverse Step 8.6 or establish superiority to the fully regularised "
                "generic representation."
            ),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_json, report)
    output_markdown.write_text(render_aggregate(report), encoding="utf-8")
    return report


def compare_authentic_to_mean_shuffled(
    authentic: Mapping[str, tuple[int, float, str]],
    shuffled_controls: Sequence[Mapping[str, tuple[int, float, str]]],
    *,
    name: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Compare authentic per-record loss with the mean loss across shuffled replicas."""

    if not shuffled_controls or not authentic:
        raise ValueError("Step 8.7 mean comparison requires authentic and shuffled predictions")
    episode_ids = sorted(authentic)
    if any(set(control) != set(authentic) for control in shuffled_controls):
        raise ValueError("Step 8.7 shuffled controls do not cover identical episodes")
    labels = [authentic[episode_id][0] for episode_id in episode_ids]
    authentic_probabilities = [authentic[episode_id][1] for episode_id in episode_ids]
    authentic_losses = per_record_log_losses(labels, authentic_probabilities)
    control_losses_by_replicate = []
    for control in shuffled_controls:
        probabilities = []
        for episode_id in episode_ids:
            target, probability, lineage = control[episode_id]
            authentic_target, _, authentic_lineage = authentic[episode_id]
            if target != authentic_target or lineage != authentic_lineage:
                raise ValueError(f"Step 8.7 metadata differs for {episode_id}")
            probabilities.append(probability)
        control_losses_by_replicate.append(per_record_log_losses(labels, probabilities))

    differences_by_lineage: dict[str, list[float]] = defaultdict(list)
    differences = []
    mean_control_losses = []
    for index, episode_id in enumerate(episode_ids):
        mean_control_loss = statistics.fmean(
            losses[index] for losses in control_losses_by_replicate
        )
        difference = authentic_losses[index] - mean_control_loss
        differences.append(difference)
        mean_control_losses.append(mean_control_loss)
        differences_by_lineage[authentic[episode_id][2]].append(difference)
    interval = lineage_bootstrap_interval(
        differences_by_lineage,
        seed=seed,
        replicates=replicates,
    )
    return {
        "name": name,
        "records": len(episode_ids),
        "lineages": len(differences_by_lineage),
        "shuffled_replicates": len(shuffled_controls),
        "mean_log_loss_difference": statistics.fmean(differences),
        "confidence_interval_95": list(interval),
        "authentic_log_loss": statistics.fmean(authentic_losses),
        "mean_shuffled_log_loss": statistics.fmean(mean_control_losses),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def render_aggregate(report: Mapping[str, Any]) -> str:
    specificity = report["authentic_preference_specificity"]
    lines = [
        "# Step 8.7 Editorial MR.Q — Shuffled-Preference Control Result",
        "",
        f"- Episodes: `{int(report['episodes']):,}`",
        f"- Shuffled replicas: `{int(report['shuffle_replicates'])}`",
        f"- Authentic preference specificity supported: `{bool(specificity['supported'])}`",
        "",
        "| Arm | Authentic log loss | Mean shuffled log loss | Difference | 95% interval | Negative replicas |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for arm in ("mrq_choice_aware", "mrq_blind"):
        comparison = report["comparisons"]["mean"][arm]
        interval = comparison["confidence_interval_95"]
        lines.append(
            f"| {arm} | {float(comparison['authentic_log_loss']):.6f} | "
            f"{float(comparison['mean_shuffled_log_loss']):.6f} | "
            f"{float(comparison['mean_log_loss_difference']):.6f} | "
            f"[{float(interval[0]):.6f}, {float(interval[1]):.6f}] | "
            f"{int(report['comparisons']['negative_point_estimate_counts'][arm])}/"
            f"{int(report['shuffle_replicates'])} |"
        )
    lines.extend(
        [
            "",
            "## Source-control diagnostics",
            "",
            "| Replica | Authentic source-test accuracy | Authentic source-test log loss |",
            "|---:|---:|---:|",
        ]
    )
    for row in report["source_controls"]:
        lines.append(
            f"| {int(row['replicate'])} | {float(row['authentic_test_accuracy']):.6f} | "
            f"{float(row['authentic_test_log_loss']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            str(specificity["claim"]),
            "",
            str(report["programme_context"]["note"]),
            "",
        ]
    )
    return "\n".join(lines)


def _pool_predictions(
    root: Path,
    arm: str,
    *,
    outer_folds: int,
    replicate: int | None = None,
) -> dict[str, tuple[int, float, str]]:
    pooled: dict[str, tuple[int, float, str]] = {}
    for fold in range(outer_folds):
        if replicate is None:
            report_path = root / "runs" / f"fold-{fold:02d}" / arm / "report.json"
        else:
            report_path = (
                root
                / "runs"
                / f"replicate-{replicate:02d}"
                / f"fold-{fold:02d}"
                / arm
                / "report.json"
            )
        report = load_canonical_report(report_path)
        prediction_path = Path(str(report["artifacts"]["predictions_path"]))
        if sha256_file(prediction_path) != str(report["artifacts"]["predictions_sha256"]):
            raise ValueError(f"Step 8.7 prediction artifact changed: {prediction_path}")
        for row in load_jsonl(prediction_path):
            if str(row.get("partition")) != "test":
                continue
            episode_id = str(row["episode_id"])
            if episode_id in pooled:
                raise ValueError(f"episode appears in multiple test folds: {episode_id}")
            pooled[episode_id] = (
                int(row["future_revised"]),
                float(row["probability_future_revised"]),
                str(row["lineage_id"]),
            )
    if not pooled:
        raise ValueError(f"Step 8.7 pooled arm is empty: {arm}")
    return pooled


def _metrics(predictions: Mapping[str, tuple[int, float, str]]) -> dict[str, Any]:
    ordered = [predictions[key] for key in sorted(predictions)]
    return binary_metrics(
        [value[0] for value in ordered],
        [value[1] for value in ordered],
    )


def _source_summaries(root: Path, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for replicate in range(int(contract["shuffle_replicates"])):
        weighted_accuracy = 0.0
        weighted_log_loss = 0.0
        records = 0
        changed_fractions = []
        for fold in range(int(contract["outer_folds"])):
            report = load_canonical_report(
                root
                / "source"
                / f"replicate-{replicate:02d}"
                / f"fold-{fold:02d}"
                / "report.json"
            )
            metrics = report["authentic_metrics"]["test"]
            count = int(metrics["records"])
            weighted_accuracy += float(metrics["accuracy"]) * count
            weighted_log_loss += float(metrics["log_loss"]) * count
            records += count
            changed_fractions.append(float(report["label_shuffle"]["train"]["changed_fraction"]))
        summaries.append(
            {
                "replicate": replicate,
                "records": records,
                "authentic_test_accuracy": weighted_accuracy / records,
                "authentic_test_log_loss": weighted_log_loss / records,
                "mean_train_label_changed_fraction": statistics.fmean(changed_fractions),
            }
        )
    return summaries
