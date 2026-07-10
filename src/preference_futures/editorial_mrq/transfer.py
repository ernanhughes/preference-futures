"""Step 8.4 future-transfer representations, identical probes, and aggregation."""

from __future__ import annotations

import math
import random
import shutil
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq.runtime import _build_mrq_model, partition_row_indices
from preference_futures.probes.common import (
    L2_GRID,
    STANDARDISATION_EPSILON,
    select_l2_candidate,
)
from preference_futures.probes.metrics import binary_metrics, per_record_log_losses
from preference_futures.probes.runtime import _fit_candidate, _require_probe_stack
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
    write_jsonl,
)
from preference_futures.training.runtime import _device_name, _resolve_device, _set_seed

TRANSFER_CONTRACT_SCHEMA_VERSION = 1
TRANSFER_RUN_SCHEMA_VERSION = 1
TRANSFER_AGGREGATE_SCHEMA_VERSION = 1
ARMS = (
    "generic_unoriented",
    "generic_choice_aware",
    "mrq_blind",
    "mrq_choice_aware",
)
PRIMARY_COMPARISON = ("mrq_choice_aware", "generic_choice_aware")
SECONDARY_COMPARISON = ("mrq_blind", "generic_unoriented")
OPTIMIZER_SETTINGS = {
    "lr": 1.0,
    "max_iter": 100,
    "max_eval": 125,
    "tolerance_grad": 1e-7,
    "tolerance_change": 1e-9,
    "history_size": 10,
    "line_search_fn": "strong_wolfe",
}
BOOTSTRAP_SEED = 17
BOOTSTRAP_REPLICATES = 10_000


def prepare_future_transfer(
    editorial_directory: Path,
    *,
    output_directory: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze Step 8.4 only after the pooled source-task gate has passed."""

    editorial_root = editorial_directory.expanduser().resolve()
    output = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else editorial_root / "future-transfer"
    )
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"Step 8.4 output is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    editorial_contract_path = editorial_root / "contract.json"
    editorial_contract = load_json(editorial_contract_path)
    source_aggregate_path = editorial_root / "rankers" / "aggregate.json"
    source_aggregate = load_json(source_aggregate_path)
    _validate_canonical_report(source_aggregate, source_aggregate_path)
    if source_aggregate.get("overall_source_gate", {}).get("passed") is not True:
        raise ValueError("Step 8.4 is blocked because the pooled source-task gate did not pass")

    embedding_report_path = editorial_root / "embeddings" / "report.json"
    embedding_report = load_json(embedding_report_path)
    tensor_path = Path(str(embedding_report["artifacts"]["embeddings_path"]))
    rows_path = Path(str(embedding_report["artifacts"]["rows_path"]))
    _require_hash(
        tensor_path,
        str(embedding_report["artifacts"]["embeddings_sha256"]),
        "Step 8 frozen embeddings",
    )
    _require_hash(
        rows_path,
        str(embedding_report["artifacts"]["rows_sha256"]),
        "Step 8 embedding rows",
    )

    outer_folds = int(editorial_contract["outer_folds"])
    fold_models = []
    for fold in range(outer_folds):
        run_directory = editorial_root / "rankers" / f"fold-{fold:02d}" / "mrq"
        report_path = run_directory / "report.json"
        report = load_json(report_path)
        _validate_canonical_report(report, report_path)
        model_path = Path(str(report["artifacts"]["model_path"]))
        _require_hash(model_path, str(report["artifacts"]["model_sha256"]), "MR.Q model")
        fold_models.append(
            {
                "fold": fold,
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "source_test_log_loss": float(report["test"]["log_loss"]),
            }
        )

    split_path = Path(str(editorial_contract["sources"]["split_manifest"]["path"]))
    contract: dict[str, Any] = {
        "step_8_transfer_contract_schema_version": TRANSFER_CONTRACT_SCHEMA_VERSION,
        "status": "frozen_before_future_probe_training",
        "exploratory": True,
        "seed": int(editorial_contract["seed"]),
        "outer_folds": outer_folds,
        "arms": list(ARMS),
        "sources": {
            "editorial_contract": _file_source(editorial_contract_path),
            "source_aggregate": _file_source(source_aggregate_path),
            "embeddings": {
                "report_path": str(embedding_report_path),
                "report_sha256": sha256_file(embedding_report_path),
                "tensor_path": str(tensor_path),
                "tensor_sha256": sha256_file(tensor_path),
                "rows_path": str(rows_path),
                "rows_sha256": sha256_file(rows_path),
            },
            "split_manifest": _file_source(split_path),
            "mrq_models": fold_models,
        },
        "target": {
            "field": "future_revised",
            "future_label_exposed_to_representation_builders": False,
            "future_label_joined_only_after_representations_are_frozen": True,
        },
        "representations": {
            "generic_unoriented": "context + mean(A,B) + abs(A-B)",
            "generic_choice_aware": "context + selected + rejected + selected-rejected",
            "mrq_blind": "h_A+h_B + abs(h_A-h_B) + abs(q_A-q_B)",
            "mrq_choice_aware": (
                "h_selected + h_rejected + h_selected-h_rejected + q_selected-q_rejected"
            ),
            "historical_choice_exposed": {
                "generic_unoriented": False,
                "generic_choice_aware": True,
                "mrq_blind": False,
                "mrq_choice_aware": True,
            },
        },
        "probe": {
            "architecture": "single_linear_logit_with_bias",
            "preprocessing": "featurewise train-only z-score",
            "l2_grid": list(L2_GRID),
            "optimizer_settings": dict(OPTIMIZER_SETTINGS),
            "selection_partition": "validation_only",
            "selection_metric": "validation_log_loss",
            "test_evaluations_per_arm_fold": 1,
        },
        "estimand": {
            "primary": {
                "comparison": "mrq_choice_aware minus generic_choice_aware test log loss",
                "negative_value_means": "MR.Q decision state improves future prediction",
            },
            "secondary": {
                "comparison": "mrq_blind minus generic_unoriented test log loss",
                "negative_value_means": "MR.Q blind state improves future prediction",
            },
            "unit": "pooled out-of-fold episode",
            "uncertainty": "paired article-lineage bootstrap",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_interval": "two-sided 95% percentile",
        },
        "gates": {
            "pooled_source_task_gate_passed": True,
            "identical_probe_architecture_for_all_arms": True,
            "identical_preprocessing_for_all_arms": True,
            "identical_l2_grid_for_all_arms": True,
            "validation_only_probe_selection": True,
            "future_labels_hidden_during_representation_building": True,
            "primary_control_receives_same_historical_choice": True,
        },
        "output_directory": str(output),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    write_json(output / "contract.json", contract)
    (output / "plan.md").write_text(render_transfer_plan(contract), encoding="utf-8")
    return contract


def run_future_transfer(
    transfer_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Build fold-specific representations and train identical future probes."""

    torch, load_file, save_file = _require_probe_stack()
    root = transfer_directory.expanduser().resolve()
    contract = _load_transfer_contract(root)
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_arms = _parse_arms(arms)
    resolved_device = _resolve_device(torch, device)

    embedding_source = contract["sources"]["embeddings"]
    tensor_path = Path(str(embedding_source["tensor_path"]))
    rows_path = Path(str(embedding_source["rows_path"]))
    _require_hash(tensor_path, str(embedding_source["tensor_sha256"]), "embedding tensor")
    _require_hash(rows_path, str(embedding_source["rows_sha256"]), "embedding rows")
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"context", "candidate_a", "candidate_b"}:
        raise ValueError("unexpected Step 8 embedding tensor keys")
    context = tensors["context"].float().contiguous()
    candidate_a = tensors["candidate_a"].float().contiguous()
    candidate_b = tensors["candidate_b"].float().contiguous()
    rows = load_jsonl(rows_path)
    if len(rows) != int(context.shape[0]):
        raise ValueError("Step 8.4 embedding row count mismatch")

    split_source = contract["sources"]["split_manifest"]
    split_path = Path(str(split_source["path"]))
    _require_hash(split_path, str(split_source["sha256"]), "split manifest")
    assignments = load_json(split_path).get("lineage_to_outer_fold")
    if not isinstance(assignments, Mapping):
        raise ValueError("split manifest has no lineage assignments")

    generic = build_generic_representations(torch, rows, context, candidate_a, candidate_b)
    future_labels = [int(bool(row["future_revised"])) for row in rows]
    output_root = root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for fold in selected_folds:
        partitions = partition_row_indices(
            rows,
            assignments,
            fold=fold,
            outer_folds=int(contract["outer_folds"]),
        )
        mrq_representations: dict[str, Any] | None = None
        for arm in selected_arms:
            output = output_root / f"fold-{fold:02d}" / arm
            if (output / "report.json").exists() and not force:
                skipped.append({"fold": fold, "arm": arm})
                continue
            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)

            if arm in generic:
                matrix = generic[arm]
            else:
                if mrq_representations is None:
                    mrq_representations = build_mrq_representations(
                        torch=torch,
                        load_file=load_file,
                        contract=contract,
                        fold=fold,
                        rows=rows,
                        context=context,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                        device=resolved_device,
                    )
                matrix = mrq_representations[arm]

            print(f"Training Step 8.4 fold {fold:02d} / {arm} ...", flush=True)
            report = _train_future_probe(
                torch=torch,
                save_file=save_file,
                contract=contract,
                matrix=matrix,
                rows=rows,
                labels=future_labels,
                partitions=partitions,
                output=output,
                fold=fold,
                arm=arm,
                device=resolved_device,
            )
            completed.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "test_log_loss": report["test"]["log_loss"],
                    "test_accuracy": report["test"]["accuracy"],
                }
            )

    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "selection": {"folds": list(selected_folds), "arms": list(selected_arms)},
        "completed": completed,
        "skipped": skipped,
        "device": str(resolved_device),
        "device_name": _device_name(torch, resolved_device),
    }
    write_json(root / "last-run-summary.json", summary)
    return summary


def aggregate_future_transfer(
    transfer_directory: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Pool all held-out future predictions and apply the transfer estimand."""

    root = transfer_directory.expanduser().resolve()
    contract = _load_transfer_contract(root)
    output_json = root / "aggregate.json"
    output_markdown = root / "aggregate.md"
    if (output_json.exists() or output_markdown.exists()) and not force:
        raise ValueError(f"Step 8.4 aggregate exists; pass --force: {output_json}")

    rows_path = Path(str(contract["sources"]["embeddings"]["rows_path"]))
    expected_records = len(load_jsonl(rows_path))
    predictions_by_arm: dict[str, dict[str, tuple[int, float, str]]] = {}
    reports_by_arm: dict[str, Any] = {}

    for arm in ARMS:
        pooled: dict[str, tuple[int, float, str]] = {}
        fold_metrics = []
        for fold in range(int(contract["outer_folds"])):
            report_path = root / "runs" / f"fold-{fold:02d}" / arm / "report.json"
            report = load_json(report_path)
            _validate_canonical_report(report, report_path)
            if report.get("contract_sha256") != contract.get("contract_sha256"):
                raise ValueError(f"Step 8.4 report contract mismatch: {report_path}")
            prediction_path = Path(str(report["artifacts"]["predictions_path"]))
            _require_hash(
                prediction_path,
                str(report["artifacts"]["predictions_sha256"]),
                "future predictions",
            )
            test_rows = [
                row for row in load_jsonl(prediction_path) if str(row.get("partition")) == "test"
            ]
            if len(test_rows) != int(report["test"]["records"]):
                raise ValueError(f"Step 8.4 test prediction count mismatch: {prediction_path}")
            for row in test_rows:
                episode_id = str(row["episode_id"])
                if episode_id in pooled:
                    raise ValueError(f"episode appears in multiple Step 8.4 test folds: {episode_id}")
                pooled[episode_id] = (
                    int(row["future_revised"]),
                    float(row["probability_future_revised"]),
                    str(row["lineage_id"]),
                )
            fold_metrics.append(report["test"])
        if len(pooled) != expected_records:
            raise ValueError(
                f"{arm} covers {len(pooled)} pooled episodes; expected {expected_records}"
            )
        ordered = [pooled[key] for key in sorted(pooled)]
        metrics = binary_metrics(
            [value[0] for value in ordered],
            [value[1] for value in ordered],
        )
        reports_by_arm[arm] = {
            "pooled_test": metrics,
            "folds": {
                "count": len(fold_metrics),
                "mean_log_loss": statistics.fmean(
                    float(value["log_loss"]) for value in fold_metrics
                ),
                "median_log_loss": statistics.median(
                    float(value["log_loss"]) for value in fold_metrics
                ),
                "mean_accuracy": statistics.fmean(
                    float(value["accuracy"]) for value in fold_metrics
                ),
            },
        }
        predictions_by_arm[arm] = pooled

    primary = paired_transfer_comparison(
        predictions_by_arm[PRIMARY_COMPARISON[0]],
        predictions_by_arm[PRIMARY_COMPARISON[1]],
        name="mrq_choice_aware_minus_generic_choice_aware",
        seed=int(contract["estimand"]["bootstrap_seed"]),
        replicates=int(contract["estimand"]["bootstrap_replicates"]),
    )
    secondary = paired_transfer_comparison(
        predictions_by_arm[SECONDARY_COMPARISON[0]],
        predictions_by_arm[SECONDARY_COMPARISON[1]],
        name="mrq_blind_minus_generic_unoriented",
        seed=int(contract["estimand"]["bootstrap_seed"]) + 1,
        replicates=int(contract["estimand"]["bootstrap_replicates"]),
    )
    transfer_supported = (
        float(primary["mean_log_loss_difference"]) < 0.0
        and float(primary["confidence_interval_95"][1]) < 0.0
    )
    report: dict[str, Any] = {
        "step_8_transfer_aggregate_schema_version": TRANSFER_AGGREGATE_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "episodes": expected_records,
        "arms": reports_by_arm,
        "comparisons": {"primary": primary, "secondary": secondary},
        "future_transfer": {
            "supported": transfer_supported,
            "primary_rule": (
                "point estimate below zero and 95% lineage-bootstrap upper bound below zero"
            ),
            "claim": (
                "MR.Q choice-aware decision state improves future prediction beyond generic "
                "choice-aware geometry"
                if transfer_supported
                else "MR.Q future-transfer improvement was not established"
            ),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_json, report)
    output_markdown.write_text(render_transfer_aggregate(report), encoding="utf-8")
    return report


def build_generic_representations(
    torch: Any,
    rows: Sequence[Mapping[str, Any]],
    context: Any,
    candidate_a: Any,
    candidate_b: Any,
) -> dict[str, Any]:
    """Build the two frozen-geometry controls without touching future labels."""

    selected_a = torch.tensor(
        [int(row["selected_index"]) == 0 for row in rows],
        dtype=torch.bool,
    ).unsqueeze(1)
    selected = torch.where(selected_a, candidate_a, candidate_b)
    rejected = torch.where(selected_a, candidate_b, candidate_a)
    return {
        "generic_unoriented": torch.cat(
            (context, (candidate_a + candidate_b) / 2.0, (candidate_a - candidate_b).abs()),
            dim=1,
        ).contiguous(),
        "generic_choice_aware": torch.cat(
            (context, selected, rejected, selected - rejected),
            dim=1,
        ).contiguous(),
    }


def build_mrq_representations(
    *,
    torch: Any,
    load_file: Any,
    contract: Mapping[str, Any],
    fold: int,
    rows: Sequence[Mapping[str, Any]],
    context: Any,
    candidate_a: Any,
    candidate_b: Any,
    device: Any,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Extract blind and historical-choice-aware states from one frozen MR.Q fold model."""

    model_source = contract["sources"]["mrq_models"][fold]
    model_path = Path(str(model_source["model_path"]))
    _require_hash(model_path, str(model_source["model_sha256"]), "MR.Q model")
    editorial_contract = load_json(Path(contract["sources"]["editorial_contract"]["path"]))
    model = _build_mrq_model(
        torch,
        embedding_size=int(context.shape[1]),
        hidden_size=int(editorial_contract["ranker"]["hidden_size"]),
        bottleneck_size=int(editorial_contract["ranker"]["bottleneck_size"]),
        dropout=float(editorial_contract["ranker"]["dropout"]),
    ).to(device)
    model.load_state_dict(load_file(str(model_path), device="cpu"))
    model.eval()

    blind_parts = []
    choice_parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(len(rows), start + batch_size)
            a = candidate_a[start:end].to(device)
            b = candidate_b[start:end].to(device)
            c = context[start:end].to(device)
            h_a, q_a = _candidate_state(model, a, b, c)
            h_b, q_b = _candidate_state(model, b, a, c)
            blind_parts.append(
                torch.cat(
                    (h_a + h_b, (h_a - h_b).abs(), (q_a - q_b).abs().unsqueeze(1)),
                    dim=1,
                ).cpu()
            )
            selected_a = torch.tensor(
                [int(row["selected_index"]) == 0 for row in rows[start:end]],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(1)
            h_selected = torch.where(selected_a, h_a, h_b)
            h_rejected = torch.where(selected_a, h_b, h_a)
            q_selected = torch.where(selected_a.squeeze(1), q_a, q_b)
            q_rejected = torch.where(selected_a.squeeze(1), q_b, q_a)
            choice_parts.append(
                torch.cat(
                    (
                        h_selected,
                        h_rejected,
                        h_selected - h_rejected,
                        (q_selected - q_rejected).unsqueeze(1),
                    ),
                    dim=1,
                ).cpu()
            )
    return {
        "mrq_blind": torch.cat(blind_parts, dim=0).contiguous(),
        "mrq_choice_aware": torch.cat(choice_parts, dim=0).contiguous(),
    }


def paired_transfer_comparison(
    treatment: Mapping[str, tuple[int, float, str]],
    control: Mapping[str, tuple[int, float, str]],
    *,
    name: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Compute paired loss differences with a lineage-clustered bootstrap."""

    if set(treatment) != set(control) or not treatment:
        raise ValueError("paired Step 8.4 arms do not cover identical episodes")
    episode_ids = sorted(treatment)
    labels = []
    treatment_probabilities = []
    control_probabilities = []
    for episode_id in episode_ids:
        target_t, probability_t, lineage_t = treatment[episode_id]
        target_c, probability_c, lineage_c = control[episode_id]
        if target_t != target_c or lineage_t != lineage_c:
            raise ValueError(f"paired Step 8.4 metadata differs for {episode_id}")
        labels.append(target_t)
        treatment_probabilities.append(probability_t)
        control_probabilities.append(probability_c)

    treatment_losses = per_record_log_losses(labels, treatment_probabilities)
    control_losses = per_record_log_losses(labels, control_probabilities)
    differences_by_lineage: dict[str, list[float]] = defaultdict(list)
    for episode_id, treatment_loss, control_loss in zip(
        episode_ids,
        treatment_losses,
        control_losses,
        strict=True,
    ):
        differences_by_lineage[treatment[episode_id][2]].append(treatment_loss - control_loss)
    differences = [value for values in differences_by_lineage.values() for value in values]
    interval = lineage_bootstrap_interval(
        differences_by_lineage,
        seed=seed,
        replicates=replicates,
    )
    treatment_metrics = binary_metrics(labels, treatment_probabilities)
    control_metrics = binary_metrics(labels, control_probabilities)
    return {
        "name": name,
        "records": len(labels),
        "lineages": len(differences_by_lineage),
        "mean_log_loss_difference": statistics.fmean(differences),
        "confidence_interval_95": list(interval),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "treatment_log_loss": treatment_metrics["log_loss"],
        "control_log_loss": control_metrics["log_loss"],
        "treatment_minus_control_accuracy": (
            float(treatment_metrics["accuracy"]) - float(control_metrics["accuracy"])
        ),
    }


def lineage_bootstrap_interval(
    differences_by_lineage: Mapping[str, Sequence[float]],
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    if not differences_by_lineage or replicates < 1:
        raise ValueError("lineage bootstrap requires data and positive replicates")
    summaries = [
        (sum(map(float, values)), len(values))
        for _, values in sorted(differences_by_lineage.items())
        if values
    ]
    if not summaries:
        raise ValueError("lineage bootstrap has no non-empty clusters")
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        total_sum = 0.0
        total_count = 0
        for _ in range(len(summaries)):
            cluster_sum, cluster_count = summaries[rng.randrange(len(summaries))]
            total_sum += cluster_sum
            total_count += cluster_count
        estimates.append(total_sum / total_count)
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def render_transfer_plan(contract: Mapping[str, Any]) -> str:
    del contract
    return "\n".join(
        [
            "# Step 8.4 Editorial MR.Q Future Transfer Plan",
            "",
            "The pooled MR.Q source-task gate passed before this contract was frozen.",
            "",
            "## Primary comparison",
            "",
            "`mrq_choice_aware` versus `generic_choice_aware` using identical future probes.",
            "Both arms receive the historical selected/rejected orientation; only the former",
            "contains the learned MR.Q decision state.",
            "",
            "## Secondary comparison",
            "",
            "`mrq_blind` versus `generic_unoriented` without exposing the historical choice.",
            "",
            "## Decision rule",
            "",
            "Future transfer is supported only when the primary pooled log-loss difference is",
            "negative and its paired lineage-bootstrap 95% interval lies entirely below zero.",
            "",
        ]
    )


def render_transfer_aggregate(report: Mapping[str, Any]) -> str:
    lines = [
        "# Step 8.4 Editorial MR.Q — Future Transfer Result",
        "",
        f"- Episodes: `{int(report['episodes']):,}`",
        f"- Future transfer supported: `{bool(report['future_transfer']['supported'])}`",
        "",
        "| Arm | Accuracy | Log loss | Brier score | ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metrics = report["arms"][arm]["pooled_test"]
        auc = metrics["roc_auc"]
        auc_text = "null" if auc is None else f"{float(auc):.6f}"
        lines.append(
            f"| {arm} | {float(metrics['accuracy']):.6f} | "
            f"{float(metrics['log_loss']):.6f} | {float(metrics['brier_score']):.6f} | "
            f"{auc_text} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for label in ("primary", "secondary"):
        comparison = report["comparisons"][label]
        interval = comparison["confidence_interval_95"]
        lines.extend(
            [
                f"### {label.title()}",
                "",
                f"- Comparison: `{comparison['name']}`",
                (
                    "- Mean treatment-minus-control log loss: "
                    f"`{float(comparison['mean_log_loss_difference']):.6f}`"
                ),
                (
                    "- Lineage-bootstrap 95% interval: "
                    f"`[{float(interval[0]):.6f}, {float(interval[1]):.6f}]`"
                ),
                (
                    "- Treatment-minus-control accuracy: "
                    f"`{float(comparison['treatment_minus_control_accuracy']):.6f}`"
                ),
                "",
            ]
        )
    lines.extend(["## Conclusion", "", report["future_transfer"]["claim"], ""])
    return "\n".join(lines)


def _train_future_probe(
    *,
    torch: Any,
    save_file: Any,
    contract: Mapping[str, Any],
    matrix: Any,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    partitions: Mapping[str, Sequence[int]],
    output: Path,
    fold: int,
    arm: str,
    device: Any,
) -> dict[str, Any]:
    _set_seed(torch, int(contract["seed"]) + fold)
    train_indices = torch.tensor(partitions["train"], dtype=torch.long)
    validation_indices = torch.tensor(partitions["validation"], dtype=torch.long)
    test_indices = torch.tensor(partitions["test"], dtype=torch.long)
    train_matrix = matrix.index_select(0, train_indices)
    validation_matrix = matrix.index_select(0, validation_indices)
    feature_mean = train_matrix.mean(dim=0)
    feature_scale = train_matrix.std(dim=0, unbiased=False)
    feature_scale = torch.where(
        feature_scale > STANDARDISATION_EPSILON,
        feature_scale,
        torch.ones_like(feature_scale),
    )
    train_standardised = ((train_matrix - feature_mean) / feature_scale).to(device)
    validation_standardised = ((validation_matrix - feature_mean) / feature_scale).to(device)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    train_targets = label_tensor.index_select(0, train_indices).to(device)
    validation_targets = label_tensor.index_select(0, validation_indices).to(device)

    candidates = []
    states: dict[float, tuple[Any, Any]] = {}
    for l2_lambda in L2_GRID:
        candidate, weight, bias = _fit_candidate(
            torch=torch,
            train_matrix=train_standardised,
            train_targets=train_targets,
            validation_matrix=validation_standardised,
            validation_targets=validation_targets,
            l2_lambda=float(l2_lambda),
            optimizer_settings=OPTIMIZER_SETTINGS,
        )
        candidates.append(candidate)
        states[float(l2_lambda)] = (weight, bias)
    selected = select_l2_candidate(candidates)
    weight, bias = states[float(selected["l2_lambda"])]

    test_matrix = matrix.index_select(0, test_indices)
    test_standardised = ((test_matrix - feature_mean) / feature_scale).to(device)
    partition_metrics: dict[str, Any] = {}
    prediction_rows = []
    for partition, indices, features in (
        ("validation", partitions["validation"], validation_standardised),
        ("test", partitions["test"], test_standardised),
    ):
        index_tensor = torch.tensor(indices, dtype=torch.long)
        partition_labels = label_tensor.index_select(0, index_tensor).int().tolist()
        with torch.inference_mode():
            probabilities = (features @ weight + bias).sigmoid().cpu().tolist()
        partition_metrics[partition] = binary_metrics(partition_labels, probabilities)
        for source_index, target, probability in zip(
            indices,
            partition_labels,
            probabilities,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "partition": partition,
                    "episode_id": str(rows[source_index]["episode_id"]),
                    "lineage_id": str(rows[source_index]["lineage_id"]),
                    "future_revised": target,
                    "probability_future_revised": probability,
                }
            )
    with torch.inference_mode():
        train_probabilities = (train_standardised @ weight + bias).sigmoid().cpu().tolist()
    partition_metrics["train"] = binary_metrics(
        train_targets.cpu().int().tolist(),
        train_probabilities,
    )

    model_path = output / "probe.safetensors"
    save_file(
        {
            "weight": weight.cpu(),
            "bias": bias.reshape(1).cpu(),
            "feature_mean": feature_mean.cpu(),
            "feature_scale": feature_scale.cpu(),
        },
        str(model_path),
        metadata={"arm": arm, "fold": str(fold), "target": "future_revised"},
    )
    prediction_path = output / "predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    report: dict[str, Any] = {
        "step_8_transfer_run_schema_version": TRANSFER_RUN_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "arm": arm,
        "representation_size": int(matrix.shape[1]),
        "train": partition_metrics["train"],
        "validation": partition_metrics["validation"],
        "test": partition_metrics["test"],
        "selected_l2_lambda": float(selected["l2_lambda"]),
        "l2_candidates": candidates,
        "artifacts": {
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "predictions_path": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output / "report.json", report)
    return report


def _candidate_state(model: Any, candidate: Any, other: Any, context: Any) -> tuple[Any, Any]:
    features = model.candidate_features(candidate, other, context)
    hidden = model.value[0](features)
    hidden = model.value[1](hidden)
    hidden = model.value[2](hidden)
    hidden = model.value[3](hidden)
    hidden = model.value[4](hidden)
    q_value = model.value[5](hidden).squeeze(-1)
    return hidden, q_value


def _load_transfer_contract(root: Path) -> dict[str, Any]:
    path = root / "contract.json"
    contract = load_json(path)
    expected = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError("Step 8.4 contract hash is invalid")
    if contract.get("status") != "frozen_before_future_probe_training":
        raise ValueError("Step 8.4 contract is not frozen")
    for source_name in ("editorial_contract", "source_aggregate", "split_manifest"):
        source = contract["sources"][source_name]
        _require_hash(Path(str(source["path"])), str(source["sha256"]), source_name)
    return contract


def _parse_arms(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ARMS
    requested = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(requested).difference(ARMS)
    if not requested or unknown:
        raise ValueError(f"unknown or empty Step 8.4 arm selection: {sorted(unknown)}")
    return requested


def _validate_canonical_report(report: Mapping[str, Any], path: Path) -> None:
    expected = str(report.get("report_sha256", ""))
    payload = dict(report)
    payload.pop("report_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError(f"canonical report hash is invalid: {path}")
    if report.get("status") != "complete":
        raise ValueError(f"report is incomplete: {path}")


def _file_source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"{label} changed or is missing: {path}")


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower]) * (1.0 - weight) + float(values[upper]) * weight
