"""Final nonlinear interaction check over combined generic and MR.Q states."""

from __future__ import annotations

import argparse
import gc
import shutil
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq.runtime import partition_row_indices
from preference_futures.editorial_mrq.shuffled_aggregate import (
    compare_authentic_to_mean_shuffled,
)
from preference_futures.editorial_mrq.shuffled_common import load_canonical_report
from preference_futures.editorial_mrq.shuffled_runtime import (
    _build_shuffled_representations,
)
from preference_futures.editorial_mrq.transfer import (
    _load_transfer_contract,
    _require_hash,
    build_generic_representations,
    build_mrq_representations,
    paired_transfer_comparison,
)
from preference_futures.probes.metrics import binary_metrics
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = 1
BASE_SEED = 81817
BOOTSTRAP_SEED = 818171
BOOTSTRAP_REPLICATES = 10_000
SHUFFLED_REPLICATES = 5
ARMS = (
    "xgb_generic_all",
    "xgb_authentic_mrq_only",
    "xgb_generic_plus_authentic_mrq",
    "xgb_generic_plus_shuffled_mrq_r00",
    "xgb_generic_plus_shuffled_mrq_r01",
    "xgb_generic_plus_shuffled_mrq_r02",
    "xgb_generic_plus_shuffled_mrq_r03",
    "xgb_generic_plus_shuffled_mrq_r04",
)
XGB_PARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "eta": 0.03,
    "max_depth": 2,
    "min_child_weight": 20.0,
    "subsample": 0.8,
    "colsample_bytree": 0.25,
    "reg_lambda": 30.0,
    "reg_alpha": 1.0,
    "gamma": 0.0,
    "max_bin": 256,
}
MAXIMUM_BOOST_ROUNDS = 1_500
EARLY_STOPPING_ROUNDS = 75


def prepare_xgboost_combined(
    transfer_directory: Path,
    *,
    output_directory: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze the final XGBoost interaction contract before any fitting."""

    transfer_root = transfer_directory.expanduser().resolve()
    transfer_contract = _load_transfer_contract(transfer_root)
    output = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else transfer_root / "xgboost-combined"
    )
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"Step 8.8 output is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    shuffled_root = transfer_root / "shuffled-mrq-control"
    shuffled_contract_path = shuffled_root / "contract.json"
    shuffled_aggregate_path = shuffled_root / "aggregate.json"
    shuffled_contract = load_json(shuffled_contract_path)
    shuffled_aggregate = load_canonical_report(shuffled_aggregate_path)
    if int(shuffled_contract.get("shuffle_replicates", -1)) != SHUFFLED_REPLICATES:
        raise ValueError("Step 8.8 requires exactly five frozen shuffled MR.Q replicas")

    source_models = []
    for replicate in range(SHUFFLED_REPLICATES):
        for fold in range(int(transfer_contract["outer_folds"])):
            report_path = (
                shuffled_root
                / "source"
                / f"replicate-{replicate:02d}"
                / f"fold-{fold:02d}"
                / "report.json"
            )
            report = load_canonical_report(report_path)
            model_path = Path(str(report["artifacts"]["model_path"]))
            _require_hash(
                model_path,
                str(report["artifacts"]["model_sha256"]),
                "shuffled MR.Q model",
            )
            source_models.append(
                {
                    "replicate": replicate,
                    "fold": fold,
                    "report_path": str(report_path),
                    "report_sha256": sha256_file(report_path),
                    "model_path": str(model_path),
                    "model_sha256": sha256_file(model_path),
                }
            )

    contract: dict[str, Any] = {
        "step_8_xgboost_combined_schema_version": SCHEMA_VERSION,
        "status": "frozen_before_xgboost_training",
        "exploratory": True,
        "seed": BASE_SEED,
        "outer_folds": int(transfer_contract["outer_folds"]),
        "unique_episodes": len(
            load_jsonl(Path(str(transfer_contract["sources"]["embeddings"]["rows_path"])))
        ),
        "arms": list(ARMS),
        "sources": {
            "transfer_contract": _file_source(transfer_root / "contract.json"),
            "editorial_contract": _file_source(
                Path(str(transfer_contract["sources"]["editorial_contract"]["path"]))
            ),
            "embedding_tensor": {
                "path": str(transfer_contract["sources"]["embeddings"]["tensor_path"]),
                "sha256": str(transfer_contract["sources"]["embeddings"]["tensor_sha256"]),
            },
            "embedding_rows": {
                "path": str(transfer_contract["sources"]["embeddings"]["rows_path"]),
                "sha256": str(transfer_contract["sources"]["embeddings"]["rows_sha256"]),
            },
            "split_manifest": dict(transfer_contract["sources"]["split_manifest"]),
            "shuffled_contract": _file_source(shuffled_contract_path),
            "shuffled_aggregate": _file_source(shuffled_aggregate_path),
            "shuffled_source_models": source_models,
        },
        "features": {
            "generic_all": "generic_unoriented + generic_choice_aware",
            "mrq_all": "mrq_blind + mrq_choice_aware",
            "generic_plus_mrq": "generic_all + mrq_all",
            "dimensions": {
                "generic_all": 5_376,
                "mrq_all": 322,
                "generic_plus_mrq": 5_698,
            },
            "future_label_exposed_during_feature_building": False,
        },
        "xgboost": {
            "parameters": dict(XGB_PARAMETERS),
            "maximum_boost_rounds": MAXIMUM_BOOST_ROUNDS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "selection_partition": "validation_only",
            "selection_metric": "logloss",
            "test_evaluations_per_arm_fold": 1,
            "class_weighting": "none",
        },
        "estimand": {
            "primary_incremental": (
                "xgb_generic_plus_authentic_mrq minus xgb_generic_all test log loss"
            ),
            "preference_specific": (
                "xgb_generic_plus_authentic_mrq minus mean of five "
                "xgb_generic_plus_shuffled_mrq controls"
            ),
            "uncertainty": "paired article-lineage bootstrap",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "decision_rule": {
            "authentic_beats_generic": "difference and 95% upper bound below zero",
            "authentic_beats_mean_shuffled": "difference and 95% upper bound below zero",
            "authentic_beats_individual_shuffles": "negative point estimate for at least 4 of 5",
            "all_conditions_required": True,
        },
        "programme_boundary": (
            "Repeated shuffled replicas are controls over the same 12,056 episodes and are not "
            "counted as additional independent observations."
        ),
        "output_directory": str(output),
        "shuffled_control_status": bool(
            shuffled_aggregate["authentic_preference_specificity"]["supported"]
        ),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    write_json(output / "contract.json", contract)
    (output / "plan.md").write_text(render_plan(contract), encoding="utf-8")
    return contract


def run_xgboost_combined(
    experiment_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Fit shallow XGBoost models over combined frozen representations."""

    torch, load_file, xgb = _require_stack()
    root = experiment_directory.expanduser().resolve()
    contract = load_contract(root)
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_arms = parse_arms(arms)
    xgb_device = resolve_xgb_device(torch, device)

    tensor_source = contract["sources"]["embedding_tensor"]
    rows_source = contract["sources"]["embedding_rows"]
    _require_hash(Path(tensor_source["path"]), tensor_source["sha256"], "embeddings")
    _require_hash(Path(rows_source["path"]), rows_source["sha256"], "embedding rows")
    tensors = load_file(str(tensor_source["path"]), device="cpu")
    context = tensors["context"].float().contiguous()
    candidate_a = tensors["candidate_a"].float().contiguous()
    candidate_b = tensors["candidate_b"].float().contiguous()
    rows = load_jsonl(Path(rows_source["path"]))
    if len(rows) != int(context.shape[0]):
        raise ValueError("Step 8.8 embedding row count mismatch")

    split_source = contract["sources"]["split_manifest"]
    assignments = load_json(Path(str(split_source["path"]))).get("lineage_to_outer_fold")
    if not isinstance(assignments, Mapping):
        raise ValueError("Step 8.8 split manifest has no lineage assignments")
    transfer_contract = load_json(Path(contract["sources"]["transfer_contract"]["path"]))
    editorial_contract = load_json(Path(contract["sources"]["editorial_contract"]["path"]))
    generic = build_generic_representations(torch, rows, context, candidate_a, candidate_b)
    generic_all = torch.cat(
        (generic["generic_unoriented"], generic["generic_choice_aware"]),
        dim=1,
    ).contiguous()
    labels = [int(bool(row["future_revised"])) for row in rows]
    completed = []
    skipped = []

    for fold in selected_folds:
        partitions = partition_row_indices(
            rows,
            assignments,
            fold=fold,
            outer_folds=int(contract["outer_folds"]),
        )
        authentic_mrq = None
        shuffled_cache: dict[int, Any] = {}
        for arm in selected_arms:
            output = root / "runs" / f"fold-{fold:02d}" / arm
            report_path = output / "report.json"
            if report_path.exists() and not force:
                skipped.append({"fold": fold, "arm": arm})
                continue
            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)

            if arm == "xgb_generic_all":
                matrix = generic_all
            elif arm in {"xgb_authentic_mrq_only", "xgb_generic_plus_authentic_mrq"}:
                if authentic_mrq is None:
                    states = build_mrq_representations(
                        torch=torch,
                        load_file=load_file,
                        contract=transfer_contract,
                        fold=fold,
                        rows=rows,
                        context=context,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                        device=torch.device("cuda" if xgb_device == "cuda" else "cpu"),
                    )
                    authentic_mrq = torch.cat(
                        (states["mrq_blind"], states["mrq_choice_aware"]),
                        dim=1,
                    ).contiguous()
                matrix = (
                    authentic_mrq
                    if arm == "xgb_authentic_mrq_only"
                    else torch.cat((generic_all, authentic_mrq), dim=1).contiguous()
                )
            else:
                replicate = shuffled_replicate_from_arm(arm)
                if replicate not in shuffled_cache:
                    source = source_model(contract, replicate=replicate, fold=fold)
                    model_path = Path(source["model_path"])
                    _require_hash(
                        model_path,
                        source["model_sha256"],
                        "Step 8.8 shuffled model",
                    )
                    states = _build_shuffled_representations(
                        torch=torch,
                        load_file=load_file,
                        model_path=model_path,
                        editorial_contract=editorial_contract,
                        rows=rows,
                        context=context,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                        device=torch.device("cuda" if xgb_device == "cuda" else "cpu"),
                    )
                    shuffled_cache[replicate] = torch.cat(
                        (
                            states["shuffled_mrq_blind"],
                            states["shuffled_mrq_choice_aware"],
                        ),
                        dim=1,
                    ).contiguous()
                matrix = torch.cat((generic_all, shuffled_cache[replicate]), dim=1).contiguous()

            print(f"Training Step 8.8 fold {fold:02d} / {arm} ...", flush=True)
            report = train_one_arm(
                xgb=xgb,
                contract=contract,
                matrix=matrix,
                rows=rows,
                labels=labels,
                partitions=partitions,
                output=output,
                fold=fold,
                arm=arm,
                device=xgb_device,
            )
            completed.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "test_log_loss": report["test"]["log_loss"],
                    "best_iteration": report["best_iteration"],
                }
            )
            if matrix is not generic_all and matrix is not authentic_mrq:
                del matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "selection": {"folds": list(selected_folds), "arms": list(selected_arms)},
        "completed": completed,
        "skipped": skipped,
        "device": xgb_device,
    }
    write_json(root / "last-run-summary.json", summary)
    return summary


def train_one_arm(
    *,
    xgb: Any,
    contract: Mapping[str, Any],
    matrix: Any,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    partitions: Mapping[str, Sequence[int]],
    output: Path,
    fold: int,
    arm: str,
    device: str,
) -> dict[str, Any]:
    """Fit one validation-stopped shallow XGBoost model and open test once."""

    import numpy as np

    train_indices = np.asarray(partitions["train"], dtype=np.int64)
    validation_indices = np.asarray(partitions["validation"], dtype=np.int64)
    test_indices = np.asarray(partitions["test"], dtype=np.int64)
    all_values = matrix.numpy()
    label_array = np.asarray(labels, dtype=np.float32)
    max_bin = int(contract["xgboost"]["parameters"]["max_bin"])
    train_data = xgb.QuantileDMatrix(
        all_values[train_indices],
        label=label_array[train_indices],
        max_bin=max_bin,
    )
    validation_data = xgb.QuantileDMatrix(
        all_values[validation_indices],
        label=label_array[validation_indices],
        max_bin=max_bin,
        ref=train_data,
    )
    parameters = dict(contract["xgboost"]["parameters"])
    parameters["device"] = device
    parameters["seed"] = int(contract["seed"]) + fold
    booster = xgb.train(
        parameters,
        train_data,
        num_boost_round=int(contract["xgboost"]["maximum_boost_rounds"]),
        evals=[(validation_data, "validation")],
        early_stopping_rounds=int(contract["xgboost"]["early_stopping_rounds"]),
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    iteration_range = (0, best_iteration + 1)
    train_probabilities = booster.predict(train_data, iteration_range=iteration_range).tolist()
    validation_probabilities = booster.predict(
        validation_data,
        iteration_range=iteration_range,
    ).tolist()

    test_data = xgb.QuantileDMatrix(
        all_values[test_indices],
        label=label_array[test_indices],
        max_bin=max_bin,
        ref=train_data,
    )
    test_probabilities = booster.predict(test_data, iteration_range=iteration_range).tolist()
    train_labels = label_array[train_indices].astype(int).tolist()
    validation_labels = label_array[validation_indices].astype(int).tolist()
    test_labels = label_array[test_indices].astype(int).tolist()
    model_path = output / "model.ubj"
    booster.save_model(model_path)
    prediction_path = output / "predictions.jsonl"
    prediction_rows = []
    for partition, indices, targets, probabilities in (
        ("validation", partitions["validation"], validation_labels, validation_probabilities),
        ("test", partitions["test"], test_labels, test_probabilities),
    ):
        for source_index, target, probability in zip(
            indices,
            targets,
            probabilities,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "partition": partition,
                    "episode_id": str(rows[source_index]["episode_id"]),
                    "lineage_id": str(rows[source_index]["lineage_id"]),
                    "future_revised": int(target),
                    "probability_future_revised": float(probability),
                }
            )
    write_jsonl(prediction_path, prediction_rows)
    report: dict[str, Any] = {
        "step_8_xgboost_run_schema_version": SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "arm": arm,
        "representation_size": int(matrix.shape[1]),
        "best_iteration": best_iteration,
        "best_validation_score": float(booster.best_score),
        "parameters": parameters,
        "train": binary_metrics(train_labels, train_probabilities),
        "validation": binary_metrics(validation_labels, validation_probabilities),
        "test": binary_metrics(test_labels, test_probabilities),
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


def aggregate_xgboost_combined(
    experiment_directory: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Pool OOF predictions and evaluate incremental authentic MR.Q interactions."""

    root = experiment_directory.expanduser().resolve()
    contract = load_contract(root)
    output_json = root / "aggregate.json"
    output_markdown = root / "aggregate.md"
    if (output_json.exists() or output_markdown.exists()) and not force:
        raise ValueError(f"Step 8.8 aggregate exists; pass --force: {output_json}")

    predictions = {
        arm: pool_predictions(root, arm, outer_folds=int(contract["outer_folds"]))
        for arm in ARMS
    }
    metrics = {arm: pooled_metrics(values) for arm, values in predictions.items()}
    authentic_arm = "xgb_generic_plus_authentic_mrq"
    generic_arm = "xgb_generic_all"
    shuffled_arms = [
        f"xgb_generic_plus_shuffled_mrq_r{replicate:02d}"
        for replicate in range(SHUFFLED_REPLICATES)
    ]
    authentic_vs_generic = paired_transfer_comparison(
        predictions[authentic_arm],
        predictions[generic_arm],
        name="xgb_generic_plus_authentic_mrq_minus_xgb_generic_all",
        seed=int(contract["estimand"]["bootstrap_seed"]),
        replicates=int(contract["estimand"]["bootstrap_replicates"]),
    )
    individual = [
        paired_transfer_comparison(
            predictions[authentic_arm],
            predictions[arm],
            name=f"{authentic_arm}_minus_{arm}",
            seed=int(contract["estimand"]["bootstrap_seed"]) + replicate + 1,
            replicates=int(contract["estimand"]["bootstrap_replicates"]),
        )
        for replicate, arm in enumerate(shuffled_arms)
    ]
    authentic_vs_mean_shuffled = compare_authentic_to_mean_shuffled(
        predictions[authentic_arm],
        [predictions[arm] for arm in shuffled_arms],
        name="xgb_authentic_augmentation_minus_mean_shuffled_augmentation",
        seed=int(contract["estimand"]["bootstrap_seed"]) + 100,
        replicates=int(contract["estimand"]["bootstrap_replicates"]),
    )
    negative_replicates = sum(
        float(comparison["mean_log_loss_difference"]) < 0.0 for comparison in individual
    )
    supported = (
        comparison_passed(authentic_vs_generic)
        and comparison_passed(authentic_vs_mean_shuffled)
        and negative_replicates >= 4
    )
    report: dict[str, Any] = {
        "step_8_xgboost_aggregate_schema_version": SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "episodes": len(predictions[generic_arm]),
        "arms": metrics,
        "comparisons": {
            "authentic_vs_generic": authentic_vs_generic,
            "authentic_vs_mean_shuffled": authentic_vs_mean_shuffled,
            "authentic_vs_individual_shuffled": individual,
            "negative_shuffled_replicates": negative_replicates,
        },
        "nonlinear_preference_interaction": {
            "supported": supported,
            "claim": (
                "Authentic MR.Q states add nonlinear future-predictive information beyond all "
                "generic geometry and shuffled-MR.Q augmentations"
                if supported
                else "Combined XGBoost did not establish an authentic preference-specific interaction"
            ),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_json, report)
    output_markdown.write_text(render_aggregate(report), encoding="utf-8")
    return report


def pool_predictions(
    root: Path,
    arm: str,
    *,
    outer_folds: int,
) -> dict[str, tuple[int, float, str]]:
    pooled = {}
    for fold in range(outer_folds):
        report_path = root / "runs" / f"fold-{fold:02d}" / arm / "report.json"
        report = load_canonical_report(report_path)
        prediction_path = Path(str(report["artifacts"]["predictions_path"]))
        _require_hash(
            prediction_path,
            str(report["artifacts"]["predictions_sha256"]),
            "Step 8.8 predictions",
        )
        for row in load_jsonl(prediction_path):
            if row.get("partition") != "test":
                continue
            episode_id = str(row["episode_id"])
            if episode_id in pooled:
                raise ValueError(f"episode appears in multiple Step 8.8 folds: {episode_id}")
            pooled[episode_id] = (
                int(row["future_revised"]),
                float(row["probability_future_revised"]),
                str(row["lineage_id"]),
            )
    if not pooled:
        raise ValueError(f"Step 8.8 pooled arm is empty: {arm}")
    return pooled


def pooled_metrics(predictions: Mapping[str, tuple[int, float, str]]) -> dict[str, Any]:
    ordered = [predictions[key] for key in sorted(predictions)]
    return binary_metrics(
        [value[0] for value in ordered],
        [value[1] for value in ordered],
    )


def comparison_passed(comparison: Mapping[str, Any]) -> bool:
    interval = comparison["confidence_interval_95"]
    return (
        float(comparison["mean_log_loss_difference"]) < 0.0
        and float(interval[1]) < 0.0
    )


def parse_arms(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ARMS
    requested = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(requested).difference(ARMS)
    if not requested or unknown:
        raise ValueError(f"unknown or empty Step 8.8 arm selection: {sorted(unknown)}")
    return requested


def shuffled_replicate_from_arm(arm: str) -> int:
    prefix = "xgb_generic_plus_shuffled_mrq_r"
    if not arm.startswith(prefix):
        raise ValueError(f"not a shuffled Step 8.8 arm: {arm}")
    replicate = int(arm.removeprefix(prefix))
    if replicate not in range(SHUFFLED_REPLICATES):
        raise ValueError(f"invalid Step 8.8 shuffled replicate: {replicate}")
    return replicate


def source_model(contract: Mapping[str, Any], *, replicate: int, fold: int) -> Mapping[str, Any]:
    for source in contract["sources"]["shuffled_source_models"]:
        if int(source["replicate"]) == replicate and int(source["fold"]) == fold:
            return source
    raise ValueError(f"missing Step 8.8 shuffled model replicate={replicate} fold={fold}")


def resolve_xgb_device(torch: Any, value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("Step 8.8 device must be auto, cpu, or cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("Step 8.8 CUDA requested but torch reports no CUDA device")
    return normalized


def load_contract(root: Path) -> dict[str, Any]:
    path = root.expanduser().resolve() / "contract.json"
    contract = load_json(path)
    expected = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError("Step 8.8 contract hash is invalid")
    if contract.get("status") != "frozen_before_xgboost_training":
        raise ValueError("Step 8.8 contract is not frozen")
    return contract


def render_plan(contract: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Step 8.8 Combined XGBoost Plan",
            "",
            f"- Unique episodes: `{int(contract['unique_episodes']):,}`",
            "- Repeated shuffled controls count as controls, not additional observations.",
            "- Model: shallow, strongly regularised histogram XGBoost.",
            "- Tree count: validation-only early stopping.",
            "",
            "## Primary question",
            "",
            "Does adding authentic MR.Q states to all generic geometry improve held-out future",
            "log loss, and does that gain exceed the same augmentation with five shuffled MR.Q",
            "states?",
            "",
            "## Decision rule",
            "",
            "All three gates must pass: authentic augmentation beats generic-only, beats the mean",
            "shuffled augmentation with a lineage-bootstrap interval below zero, and beats at least",
            "four of five individual shuffled augmentations on the point estimate.",
            "",
        ]
    )


def render_aggregate(report: Mapping[str, Any]) -> str:
    lines = [
        "# Step 8.8 Combined XGBoost Result",
        "",
        f"- Episodes: `{int(report['episodes']):,}`",
        (
            "- Nonlinear authentic-preference interaction supported: "
            f"`{bool(report['nonlinear_preference_interaction']['supported'])}`"
        ),
        "",
        "| Arm | Accuracy | Log loss | Brier score | ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metrics = report["arms"][arm]
        auc = metrics["roc_auc"]
        lines.append(
            f"| {arm} | {float(metrics['accuracy']):.6f} | "
            f"{float(metrics['log_loss']):.6f} | {float(metrics['brier_score']):.6f} | "
            f"{'null' if auc is None else f'{float(auc):.6f}'} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for label in ("authentic_vs_generic", "authentic_vs_mean_shuffled"):
        comparison = report["comparisons"][label]
        interval = comparison["confidence_interval_95"]
        lines.extend(
            [
                f"### {label.replace('_', ' ').title()}",
                "",
                f"- Difference: `{float(comparison['mean_log_loss_difference']):.6f}`",
                (
                    "- Lineage-bootstrap 95% interval: "
                    f"`[{float(interval[0]):.6f}, {float(interval[1]):.6f}]`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                "- Authentic augmentation favourable against individual shuffled replicas: "
                f"`{int(report['comparisons']['negative_shuffled_replicates'])}/5`"
            ),
            "",
            "## Conclusion",
            "",
            str(report["nonlinear_preference_interaction"]["claim"]),
            "",
        ]
    )
    return "\n".join(lines)


def _file_source(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _require_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import xgboost as xgb
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Step 8.8 requires .[train], including xgboost") from exc
    return torch, load_file, xgb


def main() -> int:
    parser = argparse.ArgumentParser(prog="preference-futures-editorial-mrq-xgboost")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--transfer-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path)
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment-dir", type=Path, required=True)
    run_parser.add_argument("--folds", default="all")
    run_parser.add_argument("--arms", default="all")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--experiment-dir", type=Path, required=True)
    aggregate_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_xgboost_combined(
            args.transfer_dir,
            output_directory=args.output_dir,
            force=args.force,
        )
        print("Step 8.8 combined XGBoost contract prepared.")
        print(f"  Episodes: {report['unique_episodes']}")
        print(f"  Output:   {report['output_directory']}")
    elif args.command == "run":
        report = run_xgboost_combined(
            args.experiment_dir,
            folds=args.folds,
            arms=args.arms,
            device=args.device,
            force=args.force,
        )
        print("Step 8.8 combined XGBoost runs complete.")
        print(f"  Completed: {len(report['completed'])}")
        print(f"  Skipped:   {len(report['skipped'])}")
    else:
        report = aggregate_xgboost_combined(args.experiment_dir, force=args.force)
        comparison = report["comparisons"]["authentic_vs_generic"]
        print("Step 8.8 combined XGBoost aggregation complete.")
        print(
            "  Authentic-minus-generic log loss: "
            f"{comparison['mean_log_loss_difference']:.6f}"
        )
        print(
            "  Nonlinear preference interaction supported: "
            f"{report['nonlinear_preference_interaction']['supported']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
