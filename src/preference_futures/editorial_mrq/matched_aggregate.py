"""Aggregate Step 8.6 matched generic controls."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq import transfer as transfer_runtime
from preference_futures.editorial_mrq.matched_common import (
    ARMS,
    SCHEMA_VERSION,
    comparison_passed,
    load_contract,
    load_report,
)
from preference_futures.probes.metrics import binary_metrics
from preference_futures.training.common import canonical_json_sha256, load_jsonl, write_json


def aggregate(matched_directory: Path, *, force: bool = False) -> dict[str, Any]:
    root = matched_directory.expanduser().resolve()
    contract = load_contract(root)
    output_json = root / "aggregate.json"
    output_markdown = root / "aggregate.md"
    if (output_json.exists() or output_markdown.exists()) and not force:
        raise ValueError(f"Step 8.6 aggregate exists; pass --force: {output_json}")

    transfer_root = Path(contract["sources"]["transfer_contract"]["path"]).parent
    predictions: dict[str, dict[str, tuple[int, float, str]]] = {}
    arm_reports: dict[str, Any] = {}
    for arm in ("generic_unoriented", "generic_choice_aware", "mrq_blind", "mrq_choice_aware"):
        predictions[arm], arm_reports[arm] = _pool(transfer_root, arm, int(contract["outer_folds"]))
    for arm in ARMS:
        predictions[arm], arm_reports[arm] = _pool(root, arm, int(contract["outer_folds"]))

    seed = int(contract["bootstrap"]["seed"])
    replicates = int(contract["bootstrap"]["replicates"])
    specs = {
        "primary_dimension": ("mrq_choice_aware", "pca_generic_choice_aware"),
        "primary_regularisation": ("mrq_choice_aware", "extended_generic_choice_aware"),
        "secondary_dimension": ("mrq_blind", "pca_generic_unoriented"),
        "secondary_regularisation": ("mrq_blind", "extended_generic_unoriented"),
        "pca_choice_vs_original": ("pca_generic_choice_aware", "generic_choice_aware"),
        "extended_choice_vs_original": (
            "extended_generic_choice_aware",
            "generic_choice_aware",
        ),
    }
    comparisons = {
        name: transfer_runtime.paired_transfer_comparison(
            predictions[treatment],
            predictions[control],
            name=f"{treatment}_minus_{control}",
            seed=seed + index,
            replicates=replicates,
        )
        for index, (name, (treatment, control)) in enumerate(specs.items())
    }
    dimension_passed = comparison_passed(comparisons["primary_dimension"])
    regularisation_passed = comparison_passed(comparisons["primary_regularisation"])
    report: dict[str, Any] = {
        "step_8_matched_control_aggregate_schema_version": SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "episodes": len(predictions["mrq_choice_aware"]),
        "arms": arm_reports,
        "comparisons": comparisons,
        "compression_and_regularisation_specificity": {
            "supported": dimension_passed and regularisation_passed,
            "dimension_control_passed": dimension_passed,
            "regularisation_control_passed": regularisation_passed,
            "authentic_preference_specificity_claim_made": False,
            "remaining_control": "identically shaped shuffled-preference MR.Q",
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_json, report)
    output_markdown.write_text(render_aggregate(report), encoding="utf-8")
    return report


def _pool(
    root: Path,
    arm: str,
    outer_folds: int,
) -> tuple[dict[str, tuple[int, float, str]], dict[str, Any]]:
    pooled: dict[str, tuple[int, float, str]] = {}
    fold_reports = []
    for fold in range(outer_folds):
        report_path = root / "runs" / f"fold-{fold:02d}" / arm / "report.json"
        report = load_report(report_path)
        prediction_path = Path(str(report["artifacts"]["predictions_path"]))
        transfer_runtime._require_hash(
            prediction_path,
            str(report["artifacts"]["predictions_sha256"]),
            f"{arm} predictions",
        )
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
        fold_reports.append(report)
    ordered = [pooled[key] for key in sorted(pooled)]
    metrics = binary_metrics(
        [value[0] for value in ordered],
        [value[1] for value in ordered],
    )
    return pooled, {
        "pooled_test": metrics,
        "folds": {
            "count": len(fold_reports),
            "selected_maximum_l2": sum(
                bool(report.get("selected_maximum_l2", False)) for report in fold_reports
            ),
            "selected_l2_values": [
                float(report["selected_l2_lambda"]) for report in fold_reports
            ],
        },
    }


def render_aggregate(report: Mapping[str, Any]) -> str:
    specificity = report["compression_and_regularisation_specificity"]
    lines = [
        "# Step 8.6 Editorial MR.Q — Matched Control Result",
        "",
        f"- Episodes: `{int(report['episodes']):,}`",
        f"- Compression and regularisation specificity supported: `{bool(specificity['supported'])}`",
        "- Authentic preference specificity claimed: `False`",
        "",
        "| Arm | Accuracy | Log loss | Brier score | ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    order = (
        "generic_unoriented",
        "generic_choice_aware",
        "extended_generic_unoriented",
        "extended_generic_choice_aware",
        "pca_generic_unoriented",
        "pca_generic_choice_aware",
        "mrq_blind",
        "mrq_choice_aware",
    )
    for arm in order:
        metrics = report["arms"][arm]["pooled_test"]
        auc = metrics["roc_auc"]
        auc_text = "null" if auc is None else f"{float(auc):.6f}"
        lines.append(
            f"| {arm} | {float(metrics['accuracy']):.6f} | "
            f"{float(metrics['log_loss']):.6f} | {float(metrics['brier_score']):.6f} | "
            f"{auc_text} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for name, comparison in report["comparisons"].items():
        interval = comparison["confidence_interval_95"]
        lines.extend(
            [
                f"### {name.replace('_', ' ').title()}",
                "",
                f"- Comparison: `{comparison['name']}`",
                f"- Mean log-loss difference: `{float(comparison['mean_log_loss_difference']):.6f}`",
                f"- Lineage-bootstrap 95% interval: `[{float(interval[0]):.6f}, {float(interval[1]):.6f}]`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            (
                "MR.Q transfer survives dimension- and regularisation-matched generic controls."
                if specificity["supported"]
                else "MR.Q transfer did not survive all matched generic controls."
            ),
            "",
            f"Remaining control: `{specificity['remaining_control']}`.",
            "",
        ]
    )
    return "\n".join(lines)
