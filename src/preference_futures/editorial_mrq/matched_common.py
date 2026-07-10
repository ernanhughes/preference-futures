"""Shared definitions for Step 8.6 matched generic controls."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq import transfer as transfer_runtime
from preference_futures.training.common import canonical_json_sha256, load_json, sha256_file

SCHEMA_VERSION = 1
ARMS = (
    "pca_generic_unoriented",
    "pca_generic_choice_aware",
    "extended_generic_unoriented",
    "extended_generic_choice_aware",
)
BASE_ARM = {
    "pca_generic_unoriented": "generic_unoriented",
    "pca_generic_choice_aware": "generic_choice_aware",
    "extended_generic_unoriented": "generic_unoriented",
    "extended_generic_choice_aware": "generic_choice_aware",
}
PCA_TARGET = {
    "pca_generic_unoriented": "mrq_blind",
    "pca_generic_choice_aware": "mrq_choice_aware",
}
EXTENDED_L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BOOTSTRAP_SEED = 41
BOOTSTRAP_REPLICATES = 10_000


def parse_arms(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ARMS
    requested = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(requested).difference(ARMS)
    if not requested or unknown:
        raise ValueError(f"unknown or empty Step 8.6 arm selection: {sorted(unknown)}")
    return requested


def load_contract(root: Path) -> dict[str, Any]:
    path = root / "contract.json"
    contract = load_json(path)
    expected = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError("Step 8.6 contract hash is invalid")
    if contract.get("status") != "frozen_before_matched_control_training":
        raise ValueError("Step 8.6 contract is not frozen")
    for source in contract["sources"].values():
        transfer_runtime._require_hash(
            Path(str(source["path"])), str(source["sha256"]), "Step 8.6 source"
        )
    return contract


def load_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    expected = str(report.get("report_sha256", ""))
    payload = dict(report)
    payload.pop("report_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError(f"canonical report hash is invalid: {path}")
    if report.get("status") != "complete":
        raise ValueError(f"report is incomplete: {path}")
    return report


def source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def comparison_passed(comparison: Mapping[str, Any]) -> bool:
    interval = comparison["confidence_interval_95"]
    return float(comparison["mean_log_loss_difference"]) < 0.0 and float(interval[1]) < 0.0


def render_plan(contract: Mapping[str, Any]) -> str:
    dimensions = contract["pca"]["target_dimensions"]
    return "\n".join(
        [
            "# Step 8.6 Editorial MR.Q — Matched Control Plan",
            "",
            "The Step 8.5 audit found 16–18× dimension differences and maximum-L2",
            "selection in every raw generic fold.",
            "",
            "## Frozen controls",
            "",
            f"- Generic unoriented PCA: `{dimensions['pca_generic_unoriented']}` dimensions",
            f"- Generic choice-aware PCA: `{dimensions['pca_generic_choice_aware']}` dimensions",
            f"- Extended L2 grid: `{contract['probe']['l2_grid']}`",
            "- PCA is fitted on training rows only and never receives future labels.",
            "",
            "## Decision rule",
            "",
            "MR.Q choice-aware must beat both the dimension-matched PCA control and the",
            "extended-L2 raw generic control with lineage-bootstrap intervals entirely below zero.",
            "",
            "A shuffled-preference MR.Q remains required for authentic-label specificity.",
            "",
        ]
    )
