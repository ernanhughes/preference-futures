"""Audit Step 8.4 representation size and regularisation before new controls."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq.transfer import ARMS, _load_transfer_contract
from preference_futures.training.common import canonical_json_sha256, load_json, write_json

SPECIFICITY_AUDIT_SCHEMA_VERSION = 1


def audit_future_transfer_specificity(
    transfer_directory: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Summarise dimensionality, selected L2 values and generalisation gaps."""

    root = transfer_directory.expanduser().resolve()
    contract = _load_transfer_contract(root)
    output_json = root / "specificity-audit.json"
    output_markdown = root / "specificity-audit.md"
    if (output_json.exists() or output_markdown.exists()) and not force:
        raise ValueError(f"Step 8.5 audit exists; pass --force: {output_json}")

    l2_grid = tuple(float(value) for value in contract["probe"]["l2_grid"])
    maximum_l2 = max(l2_grid)
    arm_summaries: dict[str, Any] = {}

    for arm in ARMS:
        fold_rows: list[dict[str, Any]] = []
        for fold in range(int(contract["outer_folds"])):
            report_path = root / "runs" / f"fold-{fold:02d}" / arm / "report.json"
            report = _load_canonical_report(report_path)
            if report.get("contract_sha256") != contract.get("contract_sha256"):
                raise ValueError(f"Step 8.5 report contract mismatch: {report_path}")
            selected_l2 = float(report["selected_l2_lambda"])
            fold_rows.append(
                {
                    "fold": fold,
                    "representation_size": int(report["representation_size"]),
                    "selected_l2_lambda": selected_l2,
                    "selected_maximum_l2": selected_l2 == maximum_l2,
                    "train_log_loss": float(report["train"]["log_loss"]),
                    "validation_log_loss": float(report["validation"]["log_loss"]),
                    "test_log_loss": float(report["test"]["log_loss"]),
                    "train_test_gap": (
                        float(report["test"]["log_loss"])
                        - float(report["train"]["log_loss"])
                    ),
                }
            )

        dimensions = sorted({int(row["representation_size"]) for row in fold_rows})
        if len(dimensions) != 1:
            raise ValueError(f"Step 8.5 arm has inconsistent dimensions: {arm}")
        l2_counts = Counter(float(row["selected_l2_lambda"]) for row in fold_rows)
        arm_summaries[arm] = {
            "representation_size": dimensions[0],
            "selected_l2_counts": {
                repr(value): int(l2_counts[value]) for value in sorted(l2_counts)
            },
            "maximum_l2": maximum_l2,
            "maximum_l2_selections": sum(
                bool(row["selected_maximum_l2"]) for row in fold_rows
            ),
            "mean_train_log_loss": statistics.fmean(
                float(row["train_log_loss"]) for row in fold_rows
            ),
            "mean_validation_log_loss": statistics.fmean(
                float(row["validation_log_loss"]) for row in fold_rows
            ),
            "mean_test_log_loss": statistics.fmean(
                float(row["test_log_loss"]) for row in fold_rows
            ),
            "mean_train_test_gap": statistics.fmean(
                float(row["train_test_gap"]) for row in fold_rows
            ),
            "folds": fold_rows,
        }

    generic_choice_size = int(arm_summaries["generic_choice_aware"]["representation_size"])
    mrq_choice_size = int(arm_summaries["mrq_choice_aware"]["representation_size"])
    generic_blind_size = int(arm_summaries["generic_unoriented"]["representation_size"])
    mrq_blind_size = int(arm_summaries["mrq_blind"]["representation_size"])
    report: dict[str, Any] = {
        "step_8_specificity_audit_schema_version": SPECIFICITY_AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "l2_grid": list(l2_grid),
        "arms": arm_summaries,
        "dimension_comparisons": {
            "generic_choice_aware_to_mrq_choice_aware_ratio": (
                generic_choice_size / mrq_choice_size
            ),
            "generic_unoriented_to_mrq_blind_ratio": (
                generic_blind_size / mrq_blind_size
            ),
        },
        "interpretation_gate": {
            "dimension_matched_controls_required": (
                generic_choice_size != mrq_choice_size
                or generic_blind_size != mrq_blind_size
            ),
            "extended_l2_diagnostic_indicated": any(
                int(summary["maximum_l2_selections"]) > 0
                for summary in arm_summaries.values()
            ),
            "note": (
                "Step 8.4 remains positive under its frozen rule, but preference-specific "
                "transfer requires dimension- and regularisation-matched controls."
            ),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_json, report)
    output_markdown.write_text(render_specificity_audit(report), encoding="utf-8")
    return report


def render_specificity_audit(report: Mapping[str, Any]) -> str:
    lines = [
        "# Step 8.5 Editorial MR.Q — Specificity Audit",
        "",
        "| Arm | Dimensions | Max-L2 selections | Mean train loss | Mean validation loss | Mean test loss | Train→test gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = report["arms"][arm]
        lines.append(
            f"| {arm} | {int(values['representation_size'])} | "
            f"{int(values['maximum_l2_selections'])}/{len(values['folds'])} | "
            f"{float(values['mean_train_log_loss']):.6f} | "
            f"{float(values['mean_validation_log_loss']):.6f} | "
            f"{float(values['mean_test_log_loss']):.6f} | "
            f"{float(values['mean_train_test_gap']):.6f} |"
        )
    dimensions = report["dimension_comparisons"]
    gate = report["interpretation_gate"]
    lines.extend(
        [
            "",
            "## Dimension comparison",
            "",
            (
                "- Generic choice-aware / MR.Q choice-aware dimension ratio: "
                f"`{float(dimensions['generic_choice_aware_to_mrq_choice_aware_ratio']):.3f}`"
            ),
            (
                "- Generic unoriented / MR.Q blind dimension ratio: "
                f"`{float(dimensions['generic_unoriented_to_mrq_blind_ratio']):.3f}`"
            ),
            "",
            "## Next controls",
            "",
            f"- Dimension-matched controls required: `{bool(gate['dimension_matched_controls_required'])}`",
            f"- Extended-L2 diagnostic indicated: `{bool(gate['extended_l2_diagnostic_indicated'])}`",
            "",
            str(gate["note"]),
            "",
        ]
    )
    return "\n".join(lines)


def _load_canonical_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    expected = str(report.get("report_sha256", ""))
    payload = dict(report)
    payload.pop("report_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError(f"Step 8.5 report hash is invalid: {path}")
    if report.get("status") != "complete":
        raise ValueError(f"Step 8.5 report is incomplete: {path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq-specificity")
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = audit_future_transfer_specificity(args.transfer_dir, force=args.force)
    print("Step 8.5 specificity audit complete.")
    for arm in ARMS:
        values = report["arms"][arm]
        print(
            f"  {arm}: dimensions={values['representation_size']}, "
            f"max_l2={values['maximum_l2_selections']}/{len(values['folds'])}, "
            f"test_log_loss={values['mean_test_log_loss']:.6f}"
        )
    print(
        "  Dimension-matched controls required: "
        f"{report['interpretation_gate']['dimension_matched_controls_required']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
