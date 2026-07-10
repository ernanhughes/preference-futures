"""Train shuffled-preference MR.Q controls and run identical future probes."""

from __future__ import annotations

import gc
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.editorial_mrq.runtime import (
    _build_mrq_model,
    partition_row_indices,
)
from preference_futures.editorial_mrq.shuffled_common import (
    SHUFFLED_ARMS,
    changed_fraction,
    load_canonical_report,
    load_contract,
    shuffled_labels_by_partition,
)
from preference_futures.editorial_mrq.transfer import (
    _candidate_state,
    _train_future_probe,
)
from preference_futures.probes.metrics import binary_metrics
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_file,
    write_json,
)
from preference_futures.training.runtime import _device_name, _resolve_device, _set_seed


def train_shuffled_source_models(
    control_directory: Path,
    *,
    replicas: str = "all",
    folds: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Train identical MR.Q models using only partition-preserving shuffled labels."""

    torch, load_file, save_file = _require_stack()
    root = control_directory.expanduser().resolve()
    contract = load_contract(root)
    editorial_contract = load_json(Path(str(contract["sources"]["editorial_contract"]["path"])))
    context, candidate_a, candidate_b, rows, assignments = _load_geometry(
        torch=torch,
        load_file=load_file,
        contract=contract,
    )
    resolved_device = _resolve_device(torch, device)
    context = context.to(resolved_device)
    candidate_a = candidate_a.to(resolved_device)
    candidate_b = candidate_b.to(resolved_device)
    selected_replicas = parse_int_selection(
        replicas,
        upper_bound=int(contract["shuffle_replicates"]),
    )
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    authentic_labels = [1 if int(row["selected_index"]) == 0 else 0 for row in rows]
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for replicate in selected_replicas:
        base_shuffle_seed = int(contract["shuffle_seeds"][replicate])
        for fold in selected_folds:
            output = root / "source" / f"replicate-{replicate:02d}" / f"fold-{fold:02d}"
            report_path = output / "report.json"
            if report_path.exists() and not force:
                skipped.append({"replicate": replicate, "fold": fold})
                continue
            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)
            partitions = partition_row_indices(
                rows,
                assignments,
                fold=fold,
                outer_folds=int(contract["outer_folds"]),
            )
            shuffle_seed = base_shuffle_seed + fold * 100_003
            shuffled_labels = shuffled_labels_by_partition(
                authentic_labels,
                partitions,
                seed=shuffle_seed,
            )
            model_seed = int(contract["seed"]) + replicate * 10_000 + fold * 1000 + 100
            print(
                f"Training Step 8.7 source replicate {replicate:02d} / fold {fold:02d} ...",
                flush=True,
            )
            report = _train_one_shuffled_source(
                torch=torch,
                save_file=save_file,
                contract=contract,
                editorial_contract=editorial_contract,
                rows=rows,
                context=context,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                authentic_labels=authentic_labels,
                shuffled_labels=shuffled_labels,
                partitions=partitions,
                output=output,
                replicate=replicate,
                fold=fold,
                shuffle_seed=shuffle_seed,
                model_seed=model_seed,
                device=resolved_device,
            )
            completed.append(
                {
                    "replicate": replicate,
                    "fold": fold,
                    "best_epoch": report["training"]["best_epoch"],
                    "shuffled_validation_log_loss": report["shuffled_metrics"]["validation"][
                        "log_loss"
                    ],
                    "authentic_test_log_loss": report["authentic_metrics"]["test"]["log_loss"],
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "selection": {"replicas": list(selected_replicas), "folds": list(selected_folds)},
        "completed": completed,
        "skipped": skipped,
        "device": str(resolved_device),
        "device_name": _device_name(torch, resolved_device),
    }
    write_json(root / "last-source-run-summary.json", summary)
    return summary


def run_shuffled_future_probes(
    control_directory: Path,
    *,
    replicas: str = "all",
    folds: str = "all",
    arms: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Extract frozen shuffled-MR.Q states and fit the Step 8.4 future probes."""

    torch, load_file, save_file = _require_stack()
    root = control_directory.expanduser().resolve()
    contract = load_contract(root)
    editorial_contract = load_json(Path(str(contract["sources"]["editorial_contract"]["path"])))
    context, candidate_a, candidate_b, rows, assignments = _load_geometry(
        torch=torch,
        load_file=load_file,
        contract=contract,
    )
    resolved_device = _resolve_device(torch, device)
    selected_replicas = parse_int_selection(
        replicas,
        upper_bound=int(contract["shuffle_replicates"]),
    )
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_arms = _parse_arms(arms)
    future_labels = [int(bool(row["future_revised"])) for row in rows]
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for replicate in selected_replicas:
        for fold in selected_folds:
            source_report_path = (
                root / "source" / f"replicate-{replicate:02d}" / f"fold-{fold:02d}" / "report.json"
            )
            source_report = load_canonical_report(source_report_path)
            if source_report.get("contract_sha256") != contract.get("contract_sha256"):
                raise ValueError(f"Step 8.7 source contract mismatch: {source_report_path}")
            model_path = Path(str(source_report["artifacts"]["model_path"]))
            if sha256_file(model_path) != str(source_report["artifacts"]["model_sha256"]):
                raise ValueError(f"Step 8.7 shuffled model changed: {model_path}")
            partitions = partition_row_indices(
                rows,
                assignments,
                fold=fold,
                outer_folds=int(contract["outer_folds"]),
            )
            representations = _build_shuffled_representations(
                torch=torch,
                load_file=load_file,
                model_path=model_path,
                editorial_contract=editorial_contract,
                rows=rows,
                context=context,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                device=resolved_device,
            )
            for arm in selected_arms:
                output = (
                    root
                    / "runs"
                    / f"replicate-{replicate:02d}"
                    / f"fold-{fold:02d}"
                    / arm
                )
                report_path = output / "report.json"
                if report_path.exists() and not force:
                    skipped.append({"replicate": replicate, "fold": fold, "arm": arm})
                    continue
                if output.exists():
                    shutil.rmtree(output)
                output.mkdir(parents=True, exist_ok=True)
                print(
                    f"Training Step 8.7 future probe replicate {replicate:02d} / "
                    f"fold {fold:02d} / {arm} ...",
                    flush=True,
                )
                report = _train_future_probe(
                    torch=torch,
                    save_file=save_file,
                    contract=contract,
                    matrix=representations[arm],
                    rows=rows,
                    labels=future_labels,
                    partitions=partitions,
                    output=output,
                    fold=fold,
                    arm=arm,
                    device=resolved_device,
                )
                report["shuffle_replicate"] = replicate
                report["source_report_path"] = str(source_report_path)
                report["source_report_sha256"] = sha256_file(source_report_path)
                report.pop("report_sha256", None)
                report["report_sha256"] = canonical_json_sha256(report)
                write_json(report_path, report)
                completed.append(
                    {
                        "replicate": replicate,
                        "fold": fold,
                        "arm": arm,
                        "test_log_loss": report["test"]["log_loss"],
                    }
                )
            del representations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "selection": {
            "replicas": list(selected_replicas),
            "folds": list(selected_folds),
            "arms": list(selected_arms),
        },
        "completed": completed,
        "skipped": skipped,
        "device": str(resolved_device),
        "device_name": _device_name(torch, resolved_device),
    }
    write_json(root / "last-transfer-run-summary.json", summary)
    return summary


def _train_one_shuffled_source(
    *,
    torch: Any,
    save_file: Any,
    contract: Mapping[str, Any],
    editorial_contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    context: Any,
    candidate_a: Any,
    candidate_b: Any,
    authentic_labels: Sequence[int],
    shuffled_labels: Sequence[int],
    partitions: Mapping[str, Sequence[int]],
    output: Path,
    replicate: int,
    fold: int,
    shuffle_seed: int,
    model_seed: int,
    device: Any,
) -> dict[str, Any]:
    _set_seed(torch, model_seed)
    settings = editorial_contract["ranker"]
    model = _build_mrq_model(
        torch,
        embedding_size=int(context.shape[1]),
        hidden_size=int(settings["hidden_size"]),
        bottleneck_size=int(settings["bottleneck_size"]),
        dropout=float(settings["dropout"]),
    ).to(device)
    shuffled_tensor = torch.tensor(shuffled_labels, dtype=torch.float32, device=device)
    authentic_tensor = torch.tensor(authentic_labels, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    maximum_epochs = int(settings["maximum_epochs"])
    batch_size = int(settings["batch_size"])
    patience = int(settings["patience"])
    best_state: dict[str, Any] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    trajectory: list[dict[str, Any]] = []
    train_indices = list(partitions["train"])
    validation_indices = torch.tensor(partitions["validation"], dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(model_seed)

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator).tolist()
        running_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            source_indices = torch.tensor(
                [train_indices[position] for position in positions],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                candidate_a.index_select(0, source_indices),
                candidate_b.index_select(0, source_indices),
                context.index_select(0, source_indices),
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                shuffled_tensor.index_select(0, source_indices),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.inference_mode():
            logits = model(
                candidate_a.index_select(0, validation_indices),
                candidate_b.index_select(0, validation_indices),
                context.index_select(0, validation_indices),
            )
            probabilities = logits.sigmoid().cpu().tolist()
            labels = shuffled_tensor.index_select(0, validation_indices).cpu().int().tolist()
            metrics = binary_metrics(labels, probabilities)
        validation_loss = float(metrics["log_loss"])
        trajectory.append(
            {
                "epoch": epoch,
                "mean_training_loss": running_loss / max(1, batches),
                "shuffled_validation_log_loss": validation_loss,
                "shuffled_validation_accuracy": metrics["accuracy"],
            }
        )
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise ValueError("Step 8.7 shuffled MR.Q produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    shuffled_metrics: dict[str, Any] = {}
    authentic_metrics: dict[str, Any] = {}
    with torch.inference_mode():
        all_logits = model(candidate_a, candidate_b, context)
        swapped_logits = model(candidate_b, candidate_a, context)
        symmetry_error = float((all_logits + swapped_logits).abs().max().cpu())
        for partition, indices in partitions.items():
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            probabilities = all_logits.index_select(0, index_tensor).sigmoid().cpu().tolist()
            shuffled_partition = shuffled_tensor.index_select(0, index_tensor).cpu().int().tolist()
            authentic_partition = authentic_tensor.index_select(0, index_tensor).cpu().int().tolist()
            shuffled_metrics[partition] = binary_metrics(shuffled_partition, probabilities)
            authentic_metrics[partition] = binary_metrics(authentic_partition, probabilities)

    model_path = output / "model.safetensors"
    save_file(
        best_state,
        str(model_path),
        metadata={
            "ranker": "shuffled_mrq",
            "candidate_order_symmetric": "true",
            "replicate": str(replicate),
            "fold": str(fold),
        },
    )
    report: dict[str, Any] = {
        "step_8_shuffled_source_schema_version": 1,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "replicate": replicate,
        "fold": fold,
        "shuffle_seed": shuffle_seed,
        "model_seed": model_seed,
        "training": {
            "best_epoch": best_epoch,
            "epochs_completed": len(trajectory),
            "selection_metric": "shuffled_validation_log_loss",
            "trajectory": trajectory,
        },
        "label_shuffle": {
            partition: {
                "records": len(indices),
                "authentic_positives": sum(authentic_labels[index] for index in indices),
                "shuffled_positives": sum(shuffled_labels[index] for index in indices),
                "changed_fraction": changed_fraction(
                    authentic_labels,
                    shuffled_labels,
                    indices,
                ),
            }
            for partition, indices in partitions.items()
        },
        "shuffled_metrics": shuffled_metrics,
        "authentic_metrics": authentic_metrics,
        "candidate_order": {
            "maximum_observed_swap_logit_error": symmetry_error,
            "passed": symmetry_error <= 1e-5,
        },
        "future_labels_accessed": False,
        "artifacts": {
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output / "report.json", report)
    return report


def _build_shuffled_representations(
    *,
    torch: Any,
    load_file: Any,
    model_path: Path,
    editorial_contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    context: Any,
    candidate_a: Any,
    candidate_b: Any,
    device: Any,
    batch_size: int = 512,
) -> dict[str, Any]:
    settings = editorial_contract["ranker"]
    model = _build_mrq_model(
        torch,
        embedding_size=int(context.shape[1]),
        hidden_size=int(settings["hidden_size"]),
        bottleneck_size=int(settings["bottleneck_size"]),
        dropout=float(settings["dropout"]),
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
        "shuffled_mrq_blind": torch.cat(blind_parts, dim=0).contiguous(),
        "shuffled_mrq_choice_aware": torch.cat(choice_parts, dim=0).contiguous(),
    }


def _load_geometry(
    *,
    torch: Any,
    load_file: Any,
    contract: Mapping[str, Any],
) -> tuple[Any, Any, Any, list[dict[str, Any]], Mapping[str, Any]]:
    tensor_path = Path(str(contract["sources"]["embedding_tensor"]["path"]))
    rows_path = Path(str(contract["sources"]["embedding_rows"]["path"]))
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"context", "candidate_a", "candidate_b"}:
        raise ValueError("unexpected Step 8.7 embedding tensor keys")
    context = tensors["context"].float().contiguous()
    candidate_a = tensors["candidate_a"].float().contiguous()
    candidate_b = tensors["candidate_b"].float().contiguous()
    rows = load_jsonl(rows_path)
    if len(rows) != int(context.shape[0]):
        raise ValueError("Step 8.7 embedding row count mismatch")
    assignments = load_json(Path(str(contract["sources"]["split_manifest"]["path"]))).get(
        "lineage_to_outer_fold"
    )
    if not isinstance(assignments, Mapping):
        raise ValueError("Step 8.7 split manifest has no lineage assignments")
    if not bool(torch.isfinite(context).all().item()):
        raise ValueError("Step 8.7 embeddings are non-finite")
    return context, candidate_a, candidate_b, rows, assignments


def _parse_arms(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return SHUFFLED_ARMS
    requested = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(requested).difference(SHUFFLED_ARMS)
    if not requested or unknown:
        raise ValueError(f"unknown or empty Step 8.7 arm selection: {sorted(unknown)}")
    return requested


def _require_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("Step 8.7 requires the training dependencies; install .[train]") from exc
    return torch, load_file, save_file
