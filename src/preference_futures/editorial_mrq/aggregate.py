"""Pool Step 8 out-of-fold ranker predictions and apply the source gate."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.probes.metrics import binary_metrics, per_record_log_losses
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)

RANKERS = ("linear", "mrq")
PRIOR_LOG_LOSS = math.log(2.0)


def aggregate_editorial_mrq(
    editorial_directory: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Pool each episode's single held-out test prediction across all outer folds."""

    root = editorial_directory.expanduser().resolve()
    contract = load_json(root / "contract.json")
    outer_folds = int(contract.get("outer_folds", 0))
    if outer_folds < 2:
        raise ValueError("Step 8 contract has an invalid outer-fold count")

    ranker_root = root / "rankers"
    output_json = ranker_root / "aggregate.json"
    output_markdown = ranker_root / "aggregate.md"
    if (output_json.exists() or output_markdown.exists()) and not force:
        raise ValueError(
            f"Step 8 aggregate output already exists; pass --force to replace it: {output_json}"
        )

    expected_rows = _expected_episode_count(root)
    ranker_reports: dict[str, dict[str, Any]] = {}
    ranker_predictions: dict[str, dict[str, tuple[int, float, str]]] = {}

    for ranker in RANKERS:
        fold_reports: list[dict[str, Any]] = []
        predictions: dict[str, tuple[int, float, str]] = {}
        maximum_symmetry_error = 0.0

        for fold in range(outer_folds):
            run_directory = ranker_root / f"fold-{fold:02d}" / ranker
            report = _load_ranker_report(run_directory / "report.json")
            if int(report.get("fold", -1)) != fold or report.get("ranker") != ranker:
                raise ValueError(f"unexpected Step 8 report identity: {run_directory}")
            if str(report.get("contract_sha256", "")) != str(contract.get("contract_sha256", "")):
                raise ValueError(f"Step 8 report contract mismatch: {run_directory}")

            prediction_path = Path(str(report["artifacts"]["predictions_path"]))
            if sha256_file(prediction_path) != str(report["artifacts"]["predictions_sha256"]):
                raise ValueError(f"Step 8 predictions changed: {prediction_path}")
            test_rows = [
                row for row in load_jsonl(prediction_path) if str(row.get("partition")) == "test"
            ]
            if len(test_rows) != int(report["test"]["records"]):
                raise ValueError(f"Step 8 test prediction count mismatch: {prediction_path}")

            for row in test_rows:
                episode_id = str(row.get("episode_id", ""))
                lineage_id = str(row.get("lineage_id", ""))
                target = int(row.get("target_a_selected", -1))
                probability = float(row.get("probability_a_selected", math.nan))
                if not episode_id or not lineage_id or target not in (0, 1):
                    raise ValueError(f"invalid Step 8 prediction row: {prediction_path}")
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"invalid Step 8 probability: {prediction_path}")
                if episode_id in predictions:
                    raise ValueError(f"episode appears in more than one test fold: {episode_id}")
                predictions[episode_id] = (target, probability, lineage_id)

            maximum_symmetry_error = max(
                maximum_symmetry_error,
                float(report["candidate_order"]["maximum_observed_swap_logit_error"]),
            )
            fold_reports.append(report)

        if len(predictions) != expected_rows:
            raise ValueError(
                f"{ranker} pooled test predictions cover {len(predictions)} episodes; "
                f"expected {expected_rows}"
            )

        ordered = [predictions[episode_id] for episode_id in sorted(predictions)]
        labels = [value[0] for value in ordered]
        probabilities = [value[1] for value in ordered]
        metrics = binary_metrics(labels, probabilities)
        correct = sum(
            int((probability >= 0.5) == bool(label))
            for label, probability in zip(labels, probabilities, strict=True)
        )
        interval = wilson_interval(correct, len(labels))
        source_gate = {
            "passed": (
                interval[0] > 0.5
                and float(metrics["log_loss"]) < PRIOR_LOG_LOSS
                and maximum_symmetry_error <= 1e-5
            ),
            "accuracy_interval_95": list(interval),
            "class_prior_accuracy": 0.5,
            "class_prior_log_loss": PRIOR_LOG_LOSS,
            "maximum_swap_logit_error": maximum_symmetry_error,
        }
        fold_accuracies = [float(report["test"]["accuracy"]) for report in fold_reports]
        fold_log_losses = [float(report["test"]["log_loss"]) for report in fold_reports]
        ranker_reports[ranker] = {
            "pooled_test": metrics,
            "source_gate": source_gate,
            "folds": {
                "count": len(fold_reports),
                "individual_gate_passes": sum(
                    bool(report["source_gate"]["passed"]) for report in fold_reports
                ),
                "accuracy_above_chance": sum(value > 0.5 for value in fold_accuracies),
                "log_loss_below_prior": sum(value < PRIOR_LOG_LOSS for value in fold_log_losses),
                "mean_accuracy": statistics.fmean(fold_accuracies),
                "median_accuracy": statistics.median(fold_accuracies),
                "minimum_accuracy": min(fold_accuracies),
                "maximum_accuracy": max(fold_accuracies),
                "mean_log_loss": statistics.fmean(fold_log_losses),
                "median_log_loss": statistics.median(fold_log_losses),
            },
        }
        ranker_predictions[ranker] = predictions

    paired = paired_comparison(ranker_predictions["linear"], ranker_predictions["mrq"])
    overall_gate = {
        "passed": bool(ranker_reports["mrq"]["source_gate"]["passed"])
        and float(paired["mrq_minus_linear_log_loss"]) < 0.0,
        "requirements": {
            "mrq_pooled_source_gate": bool(ranker_reports["mrq"]["source_gate"]["passed"]),
            "mrq_log_loss_below_linear": float(paired["mrq_minus_linear_log_loss"]) < 0.0,
        },
        "future_transfer_claim_made": False,
        "note": (
            "A passing source gate permits implementation of Step 8.4; it does not itself make "
            "a future-transfer claim."
        ),
    }

    report: dict[str, Any] = {
        "step_8_aggregate_schema_version": 1,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "outer_folds": outer_folds,
        "episodes": expected_rows,
        "rankers": ranker_reports,
        "paired_comparison": paired,
        "overall_source_gate": overall_gate,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_json, report)
    output_markdown.write_text(render_aggregate_markdown(report), encoding="utf-8")
    return report


def paired_comparison(
    linear: Mapping[str, tuple[int, float, str]],
    mrq: Mapping[str, tuple[int, float, str]],
) -> dict[str, Any]:
    """Compare paired out-of-fold probabilities for identical episodes."""

    if set(linear) != set(mrq) or not linear:
        raise ValueError("linear and MR.Q predictions do not cover identical episodes")
    episode_ids = sorted(linear)
    labels: list[int] = []
    linear_probabilities: list[float] = []
    mrq_probabilities: list[float] = []
    for episode_id in episode_ids:
        linear_target, linear_probability, linear_lineage = linear[episode_id]
        mrq_target, mrq_probability, mrq_lineage = mrq[episode_id]
        if linear_target != mrq_target or linear_lineage != mrq_lineage:
            raise ValueError(f"paired prediction metadata differs for {episode_id}")
        labels.append(linear_target)
        linear_probabilities.append(linear_probability)
        mrq_probabilities.append(mrq_probability)

    linear_losses = per_record_log_losses(labels, linear_probabilities)
    mrq_losses = per_record_log_losses(labels, mrq_probabilities)
    difference = statistics.fmean(
        mrq_loss - linear_loss
        for linear_loss, mrq_loss in zip(linear_losses, mrq_losses, strict=True)
    )
    linear_accuracy = binary_metrics(labels, linear_probabilities)["accuracy"]
    mrq_accuracy = binary_metrics(labels, mrq_probabilities)["accuracy"]
    return {
        "records": len(labels),
        "mrq_minus_linear_log_loss": difference,
        "mrq_minus_linear_accuracy": float(mrq_accuracy) - float(linear_accuracy),
        "mrq_better_log_loss": difference < 0.0,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson interval counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def render_aggregate_markdown(report: Mapping[str, Any]) -> str:
    linear = report["rankers"]["linear"]
    mrq = report["rankers"]["mrq"]
    paired = report["paired_comparison"]
    gate = report["overall_source_gate"]
    lines = [
        "# Step 8 Editorial MR.Q — Pooled Out-of-Fold Result",
        "",
        f"- Episodes: `{int(report['episodes']):,}`",
        f"- Overall source gate passed: `{bool(gate['passed'])}`",
        "- Future-transfer claim made: `False`",
        "",
        "| Ranker | Accuracy | Log loss | 95% accuracy interval | Fold gates | Pooled gate |",
        "|---|---:|---:|---|---:|---|",
    ]
    for name, values in (("linear", linear), ("mrq", mrq)):
        metrics = values["pooled_test"]
        source_gate = values["source_gate"]
        interval = source_gate["accuracy_interval_95"]
        lines.append(
            f"| {name} | {float(metrics['accuracy']):.6f} | "
            f"{float(metrics['log_loss']):.6f} | "
            f"[{float(interval[0]):.6f}, {float(interval[1]):.6f}] | "
            f"{int(values['folds']['individual_gate_passes'])}/{int(values['folds']['count'])} | "
            f"{bool(source_gate['passed'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            (
                "- MR.Q minus linear log loss: "
                f"`{float(paired['mrq_minus_linear_log_loss']):.6f}` "
                "(negative favours MR.Q)"
            ),
            (
                "- MR.Q minus linear accuracy: "
                f"`{float(paired['mrq_minus_linear_accuracy']):.6f}`"
            ),
            "",
            "A passing result permits Step 8.4 to be implemented. Future transfer remains an",
            "unmeasured question until the identical downstream probes are run.",
            "",
        ]
    )
    return "\n".join(lines)


def _expected_episode_count(root: Path) -> int:
    embedding_report = load_json(root / "embeddings" / "report.json")
    return int(embedding_report["rows"])


def _load_ranker_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    expected_hash = str(report.get("report_sha256", ""))
    payload = dict(report)
    payload.pop("report_sha256", None)
    if not expected_hash or canonical_json_sha256(payload) != expected_hash:
        raise ValueError(f"Step 8 ranker report hash is invalid: {path}")
    if report.get("status") != "complete":
        raise ValueError(f"Step 8 ranker report is incomplete: {path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq-aggregate")
    parser.add_argument("--editorial-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = aggregate_editorial_mrq(args.editorial_dir, force=args.force)
    print("Step 8 pooled out-of-fold aggregation complete.")
    for ranker in RANKERS:
        values = report["rankers"][ranker]
        print(
            f"  {ranker}: accuracy={values['pooled_test']['accuracy']:.6f}, "
            f"log_loss={values['pooled_test']['log_loss']:.6f}, "
            f"gate={values['source_gate']['passed']}"
        )
    print(f"  Overall source gate: {report['overall_source_gate']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
