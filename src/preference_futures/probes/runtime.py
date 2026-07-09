"""Runtime for Step 6 identical linear future probes."""

from __future__ import annotations

import gc
import math
import platform
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.probes.common import (
    L2_GRID,
    PROBE_RUN_SCHEMA_VERSION,
    STANDARDISATION_EPSILON,
    parse_arm_selection,
    select_l2_candidate,
)
from preference_futures.probes.contract import (
    build_probe_contract,
    validate_probe_contract,
    write_probe_contract,
)
from preference_futures.probes.metrics import binary_metrics
from preference_futures.training.common import (
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
    write_jsonl,
)
from preference_futures.training.runtime import _device_name, _resolve_device, _set_seed


def prepare_probes(
    *,
    representation_directory: Path,
    output_directory: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze the Step 6 contract before any probe is trained."""

    output = output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"probe output is not empty; pass --force to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = build_probe_contract(
        representation_directory=representation_directory,
        output_directory=output,
    )
    write_probe_contract(output, contract)
    return contract


def run_probe_jobs(
    probe_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Train selected probe jobs under the frozen Step 6 contract."""

    torch, load_file, save_file = _require_probe_stack()
    root = probe_directory.expanduser().resolve()
    contract = load_json(root / "contract.json")
    validate_probe_contract(contract)
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_arms = parse_arm_selection(arms)
    selected_keys = {(fold, arm) for fold in selected_folds for arm in selected_arms}
    jobs = [
        job
        for job in contract["jobs"]
        if (int(job["fold"]), str(job["regime"])) in selected_keys
    ]
    if len(jobs) != len(selected_keys):
        raise ValueError("selected Step 6 jobs do not match the frozen contract")

    labels = _load_future_labels(Path(contract["sources"]["episodes"]["path"]))
    resolved_device = _resolve_device(torch, device)
    _set_seed(torch, int(contract["seed"]))
    run_root = root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for job in sorted(jobs, key=lambda item: (int(item["fold"]), str(item["regime"]))):
        fold = int(job["fold"])
        regime = str(job["regime"])
        run_directory = run_root / f"fold-{fold:02d}" / regime
        if not force and _completed_run_matches(
            run_directory,
            contract_sha256=str(contract["contract_sha256"]),
            representation_run_sha256=str(job["representation_run_sha256"]),
        ):
            skipped.append({"fold": fold, "regime": regime})
            continue

        print(f"Training future probe fold {fold:02d} / {regime} ...", flush=True)
        train_matrix, train_rows, train_labels = _load_partition(
            torch=torch,
            load_file=load_file,
            artifact=job["artifacts"]["train"],
            labels=labels,
        )
        validation_matrix, validation_rows, validation_labels = _load_partition(
            torch=torch,
            load_file=load_file,
            artifact=job["artifacts"]["validation"],
            labels=labels,
        )
        feature_mean = train_matrix.mean(dim=0)
        feature_scale = train_matrix.std(dim=0, unbiased=False)
        feature_scale = torch.where(
            feature_scale > STANDARDISATION_EPSILON,
            feature_scale,
            torch.ones_like(feature_scale),
        )
        train_standardised = ((train_matrix - feature_mean) / feature_scale).to(
            resolved_device
        )
        validation_standardised = (
            (validation_matrix - feature_mean) / feature_scale
        ).to(resolved_device)
        train_targets = torch.tensor(train_labels, dtype=torch.float32, device=resolved_device)
        validation_targets = torch.tensor(
            validation_labels,
            dtype=torch.float32,
            device=resolved_device,
        )

        candidates = []
        candidate_states: dict[float, tuple[Any, Any]] = {}
        for l2_lambda in L2_GRID:
            candidate, weight, bias = _fit_candidate(
                torch=torch,
                train_matrix=train_standardised,
                train_targets=train_targets,
                validation_matrix=validation_standardised,
                validation_targets=validation_targets,
                l2_lambda=float(l2_lambda),
                optimizer_settings=contract["probe"]["optimizer_settings"],
            )
            candidates.append(candidate)
            candidate_states[float(l2_lambda)] = (weight, bias)
        selected = select_l2_candidate(candidates)
        selected_lambda = float(selected["l2_lambda"])
        selected_weight, selected_bias = candidate_states[selected_lambda]

        # The test partition is opened only after validation-only selection is complete.
        test_matrix, test_rows, test_labels = _load_partition(
            torch=torch,
            load_file=load_file,
            artifact=job["artifacts"]["test"],
            labels=labels,
        )
        test_standardised = ((test_matrix - feature_mean) / feature_scale).to(resolved_device)
        with torch.inference_mode():
            validation_logits = (
                validation_standardised @ selected_weight + selected_bias
            ).detach().cpu()
            validation_probabilities = validation_logits.sigmoid()
            test_logits = (test_standardised @ selected_weight + selected_bias).detach().cpu()
            test_probabilities = test_logits.sigmoid()

        train_prior = sum(train_labels) / len(train_labels)
        validation_metrics = binary_metrics(
            validation_labels,
            validation_probabilities.tolist(),
        )
        test_metrics = binary_metrics(test_labels, test_probabilities.tolist())
        baseline_validation = binary_metrics(
            validation_labels,
            [train_prior] * len(validation_labels),
        )
        baseline_test = binary_metrics(test_labels, [train_prior] * len(test_labels))

        report = _persist_probe_run(
            save_file=save_file,
            contract=contract,
            job=job,
            output_directory=run_directory,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            weight=selected_weight.detach().cpu(),
            bias=selected_bias.detach().cpu(),
            candidates=candidates,
            selected_lambda=selected_lambda,
            train_prior=train_prior,
            validation_rows=validation_rows,
            validation_labels=validation_labels,
            validation_logits=validation_logits.tolist(),
            validation_probabilities=validation_probabilities.tolist(),
            validation_metrics=validation_metrics,
            test_rows=test_rows,
            test_labels=test_labels,
            test_logits=test_logits.tolist(),
            test_probabilities=test_probabilities.tolist(),
            test_metrics=test_metrics,
            baseline_validation=baseline_validation,
            baseline_test=baseline_test,
            device=resolved_device,
            device_name=_device_name(torch, resolved_device),
            torch_version=torch.__version__,
            force=force,
        )
        completed.append(
            {
                "fold": fold,
                "regime": regime,
                "selected_l2_lambda": selected_lambda,
                "test_log_loss": report["test"]["log_loss"],
            }
        )

        del train_matrix, validation_matrix, test_matrix
        del train_standardised, validation_standardised, test_standardised
        del train_targets, validation_targets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "selection": {"folds": list(selected_folds), "arms": list(selected_arms)},
        "completed_jobs": completed,
        "skipped_jobs": skipped,
        "device": str(resolved_device),
        "run_root": str(run_root),
    }
    write_json(root / "last-run-summary.json", summary)
    return summary


def _fit_candidate(
    *,
    torch: Any,
    train_matrix: Any,
    train_targets: Any,
    validation_matrix: Any,
    validation_targets: Any,
    l2_lambda: float,
    optimizer_settings: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    hidden_size = int(train_matrix.shape[1])
    weight = torch.zeros(hidden_size, dtype=torch.float32, device=train_matrix.device)
    bias = torch.zeros((), dtype=torch.float32, device=train_matrix.device)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=float(optimizer_settings["lr"]),
        max_iter=int(optimizer_settings["max_iter"]),
        max_eval=int(optimizer_settings["max_eval"]),
        tolerance_grad=float(optimizer_settings["tolerance_grad"]),
        tolerance_change=float(optimizer_settings["tolerance_change"]),
        history_size=int(optimizer_settings["history_size"]),
        line_search_fn=str(optimizer_settings["line_search_fn"]),
    )
    closure_calls = 0

    def closure() -> Any:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        logits = train_matrix @ weight + bias
        predictive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            train_targets,
        )
        penalty = 0.5 * l2_lambda * weight.square().sum()
        objective = predictive_loss + penalty
        objective.backward()
        return objective

    optimizer.step(closure)
    with torch.inference_mode():
        train_logits = train_matrix @ weight + bias
        validation_logits = validation_matrix @ weight + bias
        train_probabilities = train_logits.sigmoid().detach().cpu().tolist()
        validation_probabilities = validation_logits.sigmoid().detach().cpu().tolist()
        train_labels = train_targets.detach().cpu().to(dtype=torch.int64).tolist()
        validation_labels = validation_targets.detach().cpu().to(dtype=torch.int64).tolist()
        objective = float(
            (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    train_logits,
                    train_targets,
                )
                + 0.5 * l2_lambda * weight.square().sum()
            )
            .detach()
            .cpu()
        )
    if not math.isfinite(objective):
        raise ValueError(f"non-finite Step 6 objective for L2={l2_lambda}")
    candidate = {
        "l2_lambda": l2_lambda,
        "closure_calls": closure_calls,
        "final_regularised_objective": objective,
        "train": binary_metrics(train_labels, train_probabilities),
        "validation": binary_metrics(validation_labels, validation_probabilities),
    }
    return candidate, weight.detach().clone(), bias.detach().clone()


def _load_partition(
    *,
    torch: Any,
    load_file: Any,
    artifact: Mapping[str, Any],
    labels: Mapping[str, int],
) -> tuple[Any, list[dict[str, Any]], list[int]]:
    vector_path = Path(str(artifact["representations_path"]))
    rows_path = Path(str(artifact["rows_path"]))
    if sha256_file(vector_path) != str(artifact["representations_sha256"]):
        raise ValueError(f"Step 5 representation changed: {vector_path}")
    if sha256_file(rows_path) != str(artifact["rows_sha256"]):
        raise ValueError(f"Step 5 row metadata changed: {rows_path}")
    tensors = load_file(str(vector_path), device="cpu")
    if set(tensors) != {"representations"}:
        raise ValueError(f"unexpected Step 5 tensor keys: {vector_path}")
    matrix = tensors["representations"].float().contiguous()
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"invalid Step 5 representation matrix: {vector_path}")
    rows = load_jsonl(rows_path)
    if len(rows) != int(matrix.shape[0]):
        raise ValueError(f"Step 5 row count mismatch: {rows_path}")
    partition_labels = []
    for expected_index, row in enumerate(rows):
        if int(row.get("row_index", -1)) != expected_index:
            raise ValueError(f"Step 5 row order changed: {rows_path}")
        episode_id = str(row.get("episode_id", ""))
        if episode_id not in labels:
            raise ValueError(f"unknown Step 6 episode ID: {episode_id}")
        partition_labels.append(labels[episode_id])
    return matrix, rows, partition_labels


def _load_future_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for record in load_jsonl(path):
        episode_id = str(record.get("episode_id", ""))
        target = record.get("future_revised")
        if not episode_id or episode_id in labels or type(target) is not bool:
            raise ValueError("invalid Step 6 future-label source")
        labels[episode_id] = int(target)
    if not labels:
        raise ValueError("Step 6 future-label source is empty")
    return labels


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    logits: Sequence[float],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "row_index": int(row["row_index"]),
            "episode_id": str(row["episode_id"]),
            "lineage_id": str(row["lineage_id"]),
            "future_revised": bool(label),
            "logit": float(logit),
            "probability": float(probability),
        }
        for row, label, logit, probability in zip(
            rows,
            labels,
            logits,
            probabilities,
            strict=True,
        )
    ]


def _persist_probe_run(
    *,
    save_file: Any,
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
    output_directory: Path,
    feature_mean: Any,
    feature_scale: Any,
    weight: Any,
    bias: Any,
    candidates: Sequence[Mapping[str, Any]],
    selected_lambda: float,
    train_prior: float,
    validation_rows: Sequence[Mapping[str, Any]],
    validation_labels: Sequence[int],
    validation_logits: Sequence[float],
    validation_probabilities: Sequence[float],
    validation_metrics: Mapping[str, Any],
    test_rows: Sequence[Mapping[str, Any]],
    test_labels: Sequence[int],
    test_logits: Sequence[float],
    test_probabilities: Sequence[float],
    test_metrics: Mapping[str, Any],
    baseline_validation: Mapping[str, Any],
    baseline_test: Mapping[str, Any],
    device: Any,
    device_name: str,
    torch_version: str,
    force: bool,
) -> dict[str, Any]:
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_directory.exists():
        if not force:
            raise ValueError(f"Step 6 run output already exists: {output_directory}")
        shutil.rmtree(output_directory)
    temporary.mkdir(parents=True, exist_ok=True)

    probe_path = temporary / "probe.safetensors"
    validation_path = temporary / "validation.predictions.jsonl"
    test_path = temporary / "test.predictions.jsonl"
    save_file(
        {
            "weight": weight.contiguous(),
            "bias": bias.reshape(1).contiguous(),
            "feature_mean": feature_mean.float().contiguous(),
            "feature_scale": feature_scale.float().contiguous(),
        },
        str(probe_path),
        metadata={
            "contract_sha256": str(contract["contract_sha256"]),
            "fold": str(job["fold"]),
            "regime": str(job["regime"]),
            "selected_l2_lambda": repr(selected_lambda),
        },
    )
    write_jsonl(
        validation_path,
        _prediction_rows(
            validation_rows,
            validation_labels,
            validation_logits,
            validation_probabilities,
        ),
    )
    write_jsonl(
        test_path,
        _prediction_rows(test_rows, test_labels, test_logits, test_probabilities),
    )
    report = {
        "probe_run_schema_version": PROBE_RUN_SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "fold": int(job["fold"]),
        "regime": str(job["regime"]),
        "encoder_sha256": str(job["encoder_sha256"]),
        "representation_run_sha256": str(job["representation_run_sha256"]),
        "source_task_status": job.get("source_task_status"),
        "probe": {
            "architecture": contract["probe"]["architecture"],
            "selected_l2_lambda": selected_lambda,
            "selection_metric": contract["probe"]["selection_metric"],
            "selection_partition": contract["probe"]["selection_partition"],
            "retrain_after_selection": False,
            "calibration": contract["probe"]["calibration"],
            "train_prior": train_prior,
            "candidates": list(candidates),
        },
        "validation": dict(validation_metrics),
        "test": dict(test_metrics),
        "constant_prior_baseline": {
            "validation": dict(baseline_validation),
            "test": dict(baseline_test),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
            "device": str(device),
            "device_name": device_name,
        },
        "artifacts": {
            "probe_path": "probe.safetensors",
            "probe_sha256": sha256_file(probe_path),
            "validation_predictions_path": "validation.predictions.jsonl",
            "validation_predictions_sha256": sha256_file(validation_path),
            "test_predictions_path": "test.predictions.jsonl",
            "test_predictions_sha256": sha256_file(test_path),
        },
    }
    write_json(temporary / "run.json", report)
    temporary.replace(output_directory)
    return report


def _completed_run_matches(
    run_directory: Path,
    *,
    contract_sha256: str,
    representation_run_sha256: str,
) -> bool:
    report_path = run_directory / "run.json"
    if not report_path.exists():
        return False
    try:
        report = load_json(report_path)
        artifacts = report["artifacts"]
        return (
            report.get("status") == "complete"
            and report.get("contract_sha256") == contract_sha256
            and report.get("representation_run_sha256") == representation_run_sha256
            and sha256_file(run_directory / "probe.safetensors") == artifacts["probe_sha256"]
            and sha256_file(run_directory / "validation.predictions.jsonl")
            == artifacts["validation_predictions_sha256"]
            and sha256_file(run_directory / "test.predictions.jsonl")
            == artifacts["test_predictions_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _require_probe_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "Step 6 probe dependencies are missing. Install with: "
            "python -m pip install -e '.[train]'"
        ) from exc
    return torch, load_file, save_file
