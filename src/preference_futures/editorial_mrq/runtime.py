"""Step 8 frozen semantic geometry and symmetric editorial value rankers."""

from __future__ import annotations

import gc
import math
import platform
import random
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.probes.metrics import binary_metrics
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    parse_int_selection,
    sha256_directory,
    sha256_file,
    write_json,
    write_jsonl,
)
from preference_futures.training.contract import validate_training_contract
from preference_futures.training.runtime import (
    _device_name,
    _require_training_stack,
    _resolve_device,
    _resolve_revision,
    _set_seed,
)

EDITORIAL_MRQ_CONTRACT_SCHEMA_VERSION = 1
EDITORIAL_EMBEDDING_SCHEMA_VERSION = 1
EDITORIAL_RANKER_SCHEMA_VERSION = 1
RANKER_NAMES = ("linear", "mrq")
L2_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def prepare_editorial_mrq(
    *,
    training_directory: Path,
    output_directory: Path,
    model_id: str = "sentence-transformers/all-mpnet-base-v2",
    model_revision: str = "main",
    seed: int = 17,
    maximum_sequence_length: int = 384,
    embedding_batch_size: int = 24,
    ranker_batch_size: int = 256,
    maximum_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    hidden_size: int = 256,
    bottleneck_size: int = 64,
    dropout: float = 0.1,
    teacher_weight: float = 0.25,
    force: bool = False,
) -> dict[str, Any]:
    """Snapshot one embedder and freeze the exploratory Step 8 contract."""

    training = training_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"Step 8 output is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    training_contract_path = training / "contract.json"
    training_contract = load_json(training_contract_path)
    validate_training_contract(training_contract)
    corpus_manifest_path = Path(
        str(training_contract["sources"]["step_2_manifest"]["path"])
    )
    corpus_manifest = load_json(corpus_manifest_path)
    split_source = corpus_manifest.get("sources", {}).get("split_manifest")
    if not isinstance(split_source, Mapping) or not split_source.get("path"):
        raise ValueError("Step 2 manifest has no persisted split-manifest source")
    split_manifest_path = Path(str(split_source["path"]))
    if sha256_file(split_manifest_path) != str(split_source["sha256"]):
        raise ValueError("Step 1 split manifest changed")

    episodes_path = Path(str(training_contract["sources"]["episodes"]["path"]))
    stack = _require_training_stack()
    resolved_revision = _resolve_revision(
        model_id,
        model_revision,
        model_info=stack["model_info"],
    )
    revision = None if resolved_revision == "local" else resolved_revision
    _set_seed(stack["torch"], seed)
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    model = stack["AutoModel"].from_pretrained(model_id, revision=revision)
    snapshot = output / "embedder-snapshot"
    model.save_pretrained(snapshot / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(snapshot / "tokenizer")

    contract: dict[str, Any] = {
        "editorial_mrq_contract_schema_version": EDITORIAL_MRQ_CONTRACT_SCHEMA_VERSION,
        "status": "frozen_before_step_8",
        "exploratory": True,
        "seed": seed,
        "outer_folds": int(training_contract["outer_folds"]),
        "sources": {
            "training_contract": {
                "path": str(training_contract_path),
                "sha256": sha256_file(training_contract_path),
                "contract_sha256": training_contract["contract_sha256"],
            },
            "episodes": {
                "path": str(episodes_path),
                "sha256": sha256_file(episodes_path),
            },
            "split_manifest": {
                "path": str(split_manifest_path),
                "sha256": sha256_file(split_manifest_path),
            },
        },
        "embedder": {
            "model_id": model_id,
            "requested_revision": model_revision,
            "resolved_revision": resolved_revision,
            "encoder_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "snapshot_path": str(snapshot),
            "snapshot_sha256": sha256_directory(snapshot),
            "pooling": "attention_masked_mean_then_l2_normalise",
            "maximum_sequence_length": maximum_sequence_length,
            "batch_size": embedding_batch_size,
            "frozen_during_ranker_training": True,
        },
        "ranker": {
            "names": list(RANKER_NAMES),
            "batch_size": ranker_batch_size,
            "maximum_epochs": maximum_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "hidden_size": hidden_size,
            "bottleneck_size": bottleneck_size,
            "dropout": dropout,
            "teacher_weight": teacher_weight,
            "primary_validation_metric": "binary_log_loss",
            "candidate_order_constraint": "logit(A,B) == -logit(B,A)",
        },
        "input_views": {
            "context": "context_before + context_after",
            "candidate_a": "context_before + candidate_a + context_after",
            "candidate_b": "context_before + candidate_b + context_after",
            "preference_label_exposed_to_embedder": False,
            "future_label_exposed_to_embedder": False,
        },
        "gates": {
            "parent_training_contract_valid": True,
            "parent_episode_hash_valid": True,
            "lineage_split_hash_valid": True,
            "embedder_frozen": True,
            "candidate_order_symmetry_required": True,
            "future_transfer_blocked_until_source_gate": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": stack["torch"].__version__,
            "transformers": stack["transformers_version"],
        },
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    write_json(output / "contract.json", contract)
    (output / "plan.md").write_text(render_editorial_mrq_plan(contract), encoding="utf-8")
    del model
    gc.collect()
    return contract


def run_editorial_embeddings(
    editorial_directory: Path,
    *,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Encode context and each candidate independently using the frozen embedder."""

    stack = _require_training_stack()
    torch = stack["torch"]
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Step 8 requires safetensors; install with .[train]") from exc

    root = editorial_directory.expanduser().resolve()
    contract = _load_and_validate_contract(root)
    output = root / "embeddings"
    if output.exists():
        if not force:
            raise ValueError(f"Step 8 embeddings already exist; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    episodes = load_jsonl(Path(str(contract["sources"]["episodes"]["path"])))
    rows = _validate_episode_rows(episodes)
    context_texts = [_context_view(row) for row in rows]
    candidate_a_texts = [_candidate_view(row, "candidate_a") for row in rows]
    candidate_b_texts = [_candidate_view(row, "candidate_b") for row in rows]

    snapshot = Path(str(contract["embedder"]["snapshot_path"]))
    tokenizer = stack["AutoTokenizer"].from_pretrained(snapshot / "tokenizer", use_fast=True)
    model = stack["AutoModel"].from_pretrained(snapshot / "encoder")
    resolved_device = _resolve_device(torch, device)
    model.to(resolved_device)
    model.eval()
    _set_seed(torch, int(contract["seed"]))
    encode_settings = {
        "batch_size": int(contract["embedder"]["batch_size"]),
        "maximum_length": int(contract["embedder"]["maximum_sequence_length"]),
        "device": resolved_device,
    }
    print("Encoding context views ...", flush=True)
    context = _encode_mean_pooled(
        stack=stack,
        model=model,
        tokenizer=tokenizer,
        texts=context_texts,
        **encode_settings,
    )
    print("Encoding candidate A views ...", flush=True)
    candidate_a = _encode_mean_pooled(
        stack=stack,
        model=model,
        tokenizer=tokenizer,
        texts=candidate_a_texts,
        **encode_settings,
    )
    print("Encoding candidate B views ...", flush=True)
    candidate_b = _encode_mean_pooled(
        stack=stack,
        model=model,
        tokenizer=tokenizer,
        texts=candidate_b_texts,
        **encode_settings,
    )
    if context.shape != candidate_a.shape or context.shape != candidate_b.shape:
        raise ValueError("Step 8 embedding view shapes differ")

    tensor_path = output / "embeddings.safetensors"
    save_file(
        {
            "context": context.contiguous(),
            "candidate_a": candidate_a.contiguous(),
            "candidate_b": candidate_b.contiguous(),
        },
        str(tensor_path),
        metadata={
            "contract_sha256": str(contract["contract_sha256"]),
            "pooling": str(contract["embedder"]["pooling"]),
            "future_fields_exposed_to_embedder": "false",
            "preference_label_exposed_to_embedder": "false",
        },
    )
    row_path = output / "rows.jsonl"
    write_jsonl(
        row_path,
        [
            {
                "row_index": index,
                "episode_id": str(row["episode_id"]),
                "lineage_id": str(row["lineage_id"]),
                "selected_index": int(row["selected_index"]),
                "future_revised": bool(row["future_revised"]),
            }
            for index, row in enumerate(rows)
        ],
    )
    report: dict[str, Any] = {
        "editorial_embedding_schema_version": EDITORIAL_EMBEDDING_SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "rows": len(rows),
        "hidden_size": int(context.shape[1]),
        "shape": [int(context.shape[0]), int(context.shape[1])],
        "device": str(resolved_device),
        "device_name": _device_name(torch, resolved_device),
        "artifacts": {
            "embeddings_path": str(tensor_path),
            "embeddings_sha256": sha256_file(tensor_path),
            "rows_path": str(row_path),
            "rows_sha256": sha256_file(row_path),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output / "report.json", report)
    del model, context, candidate_a, candidate_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_editorial_rankers(
    editorial_directory: Path,
    *,
    folds: str = "all",
    rankers: str = "all",
    teacher_predictions_path: Path | None = None,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Train fold-locked symmetric linear and tiny MR.Q preference rankers."""

    torch, load_file, save_file = _require_ranker_stack()
    root = editorial_directory.expanduser().resolve()
    contract = _load_and_validate_contract(root)
    embedding_report = load_json(root / "embeddings" / "report.json")
    tensor_path = Path(str(embedding_report["artifacts"]["embeddings_path"]))
    row_path = Path(str(embedding_report["artifacts"]["rows_path"]))
    if sha256_file(tensor_path) != str(embedding_report["artifacts"]["embeddings_sha256"]):
        raise ValueError("Step 8 embeddings changed")
    if sha256_file(row_path) != str(embedding_report["artifacts"]["rows_sha256"]):
        raise ValueError("Step 8 embedding rows changed")
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"context", "candidate_a", "candidate_b"}:
        raise ValueError("unexpected Step 8 embedding tensor keys")
    context = tensors["context"].float().contiguous()
    candidate_a = tensors["candidate_a"].float().contiguous()
    candidate_b = tensors["candidate_b"].float().contiguous()
    rows = load_jsonl(row_path)
    if len(rows) != context.shape[0]:
        raise ValueError("Step 8 embedding row count mismatch")

    split_manifest = load_json(Path(str(contract["sources"]["split_manifest"]["path"])))
    assignments = split_manifest.get("lineage_to_outer_fold")
    if not isinstance(assignments, Mapping):
        raise ValueError("split manifest has no lineage assignments")
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_rankers = _parse_rankers(rankers)
    teacher = _load_teacher_probabilities(teacher_predictions_path)
    resolved_device = _resolve_device(torch, device)
    output_root = root / "rankers"
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
        for ranker_name in selected_rankers:
            output = output_root / f"fold-{fold:02d}" / ranker_name
            report_path = output / "report.json"
            if report_path.exists() and not force:
                skipped.append({"fold": fold, "ranker": ranker_name})
                continue
            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)
            seed = int(contract["seed"]) + fold * 1000 + (0 if ranker_name == "linear" else 100)
            print(f"Training Step 8 fold {fold:02d} / {ranker_name} ...", flush=True)
            if ranker_name == "linear":
                report = _train_linear_ranker(
                    torch=torch,
                    save_file=save_file,
                    contract=contract,
                    rows=rows,
                    context=context,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    partitions=partitions,
                    output=output,
                    fold=fold,
                    seed=seed,
                    device=resolved_device,
                )
            else:
                report = _train_mrq_ranker(
                    torch=torch,
                    save_file=save_file,
                    contract=contract,
                    rows=rows,
                    context=context,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    partitions=partitions,
                    teacher=teacher,
                    output=output,
                    fold=fold,
                    seed=seed,
                    device=resolved_device,
                )
            completed.append(
                {
                    "fold": fold,
                    "ranker": ranker_name,
                    "test_accuracy": report["test"]["accuracy"],
                    "test_log_loss": report["test"]["log_loss"],
                    "source_gate_passed": report["source_gate"]["passed"],
                }
            )
    summary = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "selection": {"folds": list(selected_folds), "rankers": list(selected_rankers)},
        "teacher_predictions_supplied": teacher_predictions_path is not None,
        "completed": completed,
        "skipped": skipped,
        "device": str(resolved_device),
    }
    write_json(output_root / "last-run-summary.json", summary)
    return summary


def partition_row_indices(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Any],
    *,
    fold: int,
    outer_folds: int,
) -> dict[str, list[int]]:
    """Apply the original lineage-grouped train/validation/test policy."""

    validation_bucket = (fold + 1) % outer_folds
    result = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(rows):
        lineage_id = str(row["lineage_id"])
        if lineage_id not in assignments:
            raise ValueError(f"missing split assignment for lineage {lineage_id}")
        bucket = int(assignments[lineage_id])
        if bucket == fold:
            partition = "test"
        elif bucket == validation_bucket:
            partition = "validation"
        else:
            partition = "train"
        result[partition].append(index)
    if any(not values for values in result.values()):
        raise ValueError(f"fold {fold} has an empty Step 8 partition")
    return result


def antisymmetric_linear_features(torch: Any, a: Any, b: Any, context: Any) -> Any:
    """Return features that negate exactly when candidate order is reversed."""

    return torch.cat(
        (
            a - b,
            (a - context).square() - (b - context).square(),
            a * context - b * context,
        ),
        dim=1,
    )


def render_editorial_mrq_plan(contract: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Step 8 Editorial MR.Q Plan",
            "",
            "Step 8 is exploratory and does not mutate the frozen Steps 1-7 artifacts.",
            "",
            "## Frozen geometry",
            "",
            f"- Embedder: `{contract['embedder']['model_id']}`",
            f"- Revision: `{contract['embedder']['resolved_revision']}`",
            f"- Pooling: `{contract['embedder']['pooling']}`",
            f"- Maximum sequence length: `{contract['embedder']['maximum_sequence_length']}`",
            "- Encoder updates during preference training: `0`",
            "",
            "## Rankers",
            "",
            "- `linear`: antisymmetric logistic value difference over frozen embeddings.",
            "- `mrq`: shared tiny value network; probability derives from `q_A - q_B`.",
            "",
            "Both architectures guarantee that swapping A and B reverses the preference logit.",
            "Future-transfer testing remains blocked until a source-task gate passes.",
            "",
        ]
    )


def _train_linear_ranker(
    *,
    torch: Any,
    save_file: Any,
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    context: Any,
    candidate_a: Any,
    candidate_b: Any,
    partitions: Mapping[str, Sequence[int]],
    output: Path,
    fold: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    _set_seed(torch, seed)
    features = antisymmetric_linear_features(torch, candidate_a, candidate_b, context)
    labels = torch.tensor(
        [1.0 if int(row["selected_index"]) == 0 else 0.0 for row in rows],
        dtype=torch.float32,
    )
    train_indices = torch.tensor(partitions["train"], dtype=torch.long)
    scale = features.index_select(0, train_indices).std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    features = (features / scale).to(device)
    labels = labels.to(device)
    validation_indices = torch.tensor(partitions["validation"], dtype=torch.long, device=device)
    train_indices = train_indices.to(device)
    candidates: list[dict[str, Any]] = []
    states: dict[float, Any] = {}
    for l2_lambda in L2_GRID:
        weight = torch.zeros(features.shape[1], dtype=torch.float32, device=device)
        weight.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [weight],
            lr=1.0,
            max_iter=120,
            max_eval=150,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Any:
            optimizer.zero_grad(set_to_none=True)
            logits = features.index_select(0, train_indices) @ weight
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.index_select(0, train_indices),
            ) + 0.5 * float(l2_lambda) * weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.inference_mode():
            validation_logits = features.index_select(0, validation_indices) @ weight
            validation_probabilities = validation_logits.sigmoid().cpu().tolist()
            validation_labels = labels.index_select(0, validation_indices).cpu().int().tolist()
            metrics = binary_metrics(validation_labels, validation_probabilities)
        candidates.append({"l2_lambda": l2_lambda, "validation": metrics})
        states[l2_lambda] = weight.detach().cpu().clone()
    selected = min(
        candidates,
        key=lambda item: (float(item["validation"]["log_loss"]), float(item["l2_lambda"])),
    )
    weight = states[float(selected["l2_lambda"])].to(device)
    partition_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        all_logits = features @ weight
        swapped_features = antisymmetric_linear_features(
            torch,
            candidate_b.to(device),
            candidate_a.to(device),
            context.to(device),
        ) / scale.to(device)
        swapped_logits = swapped_features @ weight
        symmetry_error = float((all_logits + swapped_logits).abs().max().cpu())
        for partition, indices in partitions.items():
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            probabilities = all_logits.index_select(0, index_tensor).sigmoid().cpu().tolist()
            partition_labels = labels.index_select(0, index_tensor).cpu().int().tolist()
            partition_metrics[partition] = binary_metrics(partition_labels, probabilities)
            if partition in {"validation", "test"}:
                for source_index, probability, target in zip(
                    indices,
                    probabilities,
                    partition_labels,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "partition": partition,
                            "episode_id": str(rows[source_index]["episode_id"]),
                            "lineage_id": str(rows[source_index]["lineage_id"]),
                            "target_a_selected": target,
                            "probability_a_selected": probability,
                        }
                    )
    source_gate = _source_gate(partition_metrics["test"])
    artifact_path = output / "model.safetensors"
    save_file(
        {"weight": weight.cpu(), "feature_scale": scale.cpu()},
        str(artifact_path),
        metadata={"ranker": "linear", "candidate_order_symmetric": "true"},
    )
    return _persist_ranker_report(
        output=output,
        contract=contract,
        fold=fold,
        ranker="linear",
        seed=seed,
        train=partition_metrics["train"],
        validation=partition_metrics["validation"],
        test=partition_metrics["test"],
        source_gate=source_gate,
        symmetry_error=symmetry_error,
        artifact_path=artifact_path,
        predictions=prediction_rows,
        extra={"l2_candidates": candidates, "selected_l2_lambda": selected["l2_lambda"]},
    )


def _train_mrq_ranker(
    *,
    torch: Any,
    save_file: Any,
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    context: Any,
    candidate_a: Any,
    candidate_b: Any,
    partitions: Mapping[str, Sequence[int]],
    teacher: Mapping[str, float],
    output: Path,
    fold: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    _set_seed(torch, seed)
    model = _build_mrq_model(
        torch,
        embedding_size=int(context.shape[1]),
        hidden_size=int(contract["ranker"]["hidden_size"]),
        bottleneck_size=int(contract["ranker"]["bottleneck_size"]),
        dropout=float(contract["ranker"]["dropout"]),
    ).to(device)
    labels = torch.tensor(
        [1.0 if int(row["selected_index"]) == 0 else 0.0 for row in rows],
        dtype=torch.float32,
        device=device,
    )
    teacher_targets = torch.tensor(
        [float(teacher.get(str(row["episode_id"]), math.nan)) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    context = context.to(device)
    candidate_a = candidate_a.to(device)
    candidate_b = candidate_b.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["ranker"]["learning_rate"]),
        weight_decay=float(contract["ranker"]["weight_decay"]),
    )
    maximum_epochs = int(contract["ranker"]["maximum_epochs"])
    batch_size = int(contract["ranker"]["batch_size"])
    patience = int(contract["ranker"]["patience"])
    teacher_weight = float(contract["ranker"]["teacher_weight"])
    best_state: dict[str, Any] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    trajectory: list[dict[str, Any]] = []
    train_indices_list = list(partitions["train"])
    validation_indices = torch.tensor(partitions["validation"], dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices_list), generator=generator).tolist()
        running_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            source_indices = torch.tensor(
                [train_indices_list[position] for position in positions],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                candidate_a.index_select(0, source_indices),
                candidate_b.index_select(0, source_indices),
                context.index_select(0, source_indices),
            )
            authentic_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.index_select(0, source_indices),
            )
            batch_teacher = teacher_targets.index_select(0, source_indices)
            teacher_mask = torch.isfinite(batch_teacher)
            if teacher_weight > 0.0 and bool(teacher_mask.any().item()):
                teacher_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[teacher_mask],
                    batch_teacher[teacher_mask],
                )
                loss = (1.0 - teacher_weight) * authentic_loss + teacher_weight * teacher_loss
            else:
                teacher_loss = None
                loss = authentic_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.inference_mode():
            validation_logits = model(
                candidate_a.index_select(0, validation_indices),
                candidate_b.index_select(0, validation_indices),
                context.index_select(0, validation_indices),
            )
            validation_probabilities = validation_logits.sigmoid().cpu().tolist()
            validation_labels = labels.index_select(0, validation_indices).cpu().int().tolist()
            validation_metrics = binary_metrics(validation_labels, validation_probabilities)
        validation_loss = float(validation_metrics["log_loss"])
        trajectory.append(
            {
                "epoch": epoch,
                "mean_training_loss": running_loss / max(1, batches),
                "validation_log_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
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
        raise ValueError("MR.Q training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    partition_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        all_logits = model(candidate_a, candidate_b, context)
        swapped_logits = model(candidate_b, candidate_a, context)
        symmetry_error = float((all_logits + swapped_logits).abs().max().cpu())
        for partition, indices in partitions.items():
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            probabilities = all_logits.index_select(0, index_tensor).sigmoid().cpu().tolist()
            partition_labels = labels.index_select(0, index_tensor).cpu().int().tolist()
            partition_metrics[partition] = binary_metrics(partition_labels, probabilities)
            if partition in {"validation", "test"}:
                for source_index, probability, target in zip(
                    indices,
                    probabilities,
                    partition_labels,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "partition": partition,
                            "episode_id": str(rows[source_index]["episode_id"]),
                            "lineage_id": str(rows[source_index]["lineage_id"]),
                            "target_a_selected": target,
                            "probability_a_selected": probability,
                        }
                    )
    source_gate = _source_gate(partition_metrics["test"])
    artifact_path = output / "model.safetensors"
    save_file(
        best_state,
        str(artifact_path),
        metadata={"ranker": "mrq", "candidate_order_symmetric": "true"},
    )
    return _persist_ranker_report(
        output=output,
        contract=contract,
        fold=fold,
        ranker="mrq",
        seed=seed,
        train=partition_metrics["train"],
        validation=partition_metrics["validation"],
        test=partition_metrics["test"],
        source_gate=source_gate,
        symmetry_error=symmetry_error,
        artifact_path=artifact_path,
        predictions=prediction_rows,
        extra={
            "best_epoch": best_epoch,
            "epochs_completed": len(trajectory),
            "trajectory": trajectory,
            "teacher_examples_available": sum(
                str(row["episode_id"]) in teacher for row in rows
            ),
            "teacher_weight": teacher_weight,
        },
    )


def _build_mrq_model(
    torch: Any,
    *,
    embedding_size: int,
    hidden_size: int,
    bottleneck_size: int,
    dropout: float,
) -> Any:
    input_size = embedding_size * 6

    class EditorialMRQ(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, bottleneck_size),
                torch.nn.GELU(),
                torch.nn.Linear(bottleneck_size, 1),
            )

        def candidate_features(self, candidate: Any, other: Any, context: Any) -> Any:
            return torch.cat(
                (
                    candidate,
                    context,
                    candidate - context,
                    candidate * context,
                    candidate - other,
                    (candidate - other).abs(),
                ),
                dim=1,
            )

        def forward(self, a: Any, b: Any, context: Any) -> Any:
            q_a = self.value(self.candidate_features(a, b, context)).squeeze(-1)
            q_b = self.value(self.candidate_features(b, a, context)).squeeze(-1)
            return q_a - q_b

    return EditorialMRQ()


def _persist_ranker_report(
    *,
    output: Path,
    contract: Mapping[str, Any],
    fold: int,
    ranker: str,
    seed: int,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    source_gate: Mapping[str, Any],
    symmetry_error: float,
    artifact_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_path = output / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    report: dict[str, Any] = {
        "editorial_ranker_schema_version": EDITORIAL_RANKER_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "ranker": ranker,
        "seed": seed,
        "train": dict(train),
        "validation": dict(validation),
        "test": dict(test),
        "source_gate": dict(source_gate),
        "candidate_order": {
            "architecture_guarantees_logit_reversal": True,
            "maximum_observed_swap_logit_error": symmetry_error,
            "passed": symmetry_error <= 1e-5,
        },
        "artifacts": {
            "model_path": str(artifact_path),
            "model_sha256": sha256_file(artifact_path),
            "predictions_path": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
        },
        **dict(extra),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_ranker_markdown(report), encoding="utf-8")
    return report


def render_ranker_markdown(report: Mapping[str, Any]) -> str:
    gate = report["source_gate"]
    return "\n".join(
        [
            f"# Step 8 Editorial MR.Q — Fold {int(report['fold'])} / {report['ranker']}",
            "",
            f"- Source gate passed: `{gate['passed']}`",
            f"- Test accuracy: `{float(report['test']['accuracy']):.6f}`",
            f"- Test log loss: `{float(report['test']['log_loss']):.6f}`",
            (
                "- Test accuracy interval: "
                f"`[{float(gate['accuracy_interval_95'][0]):.6f}, "
                f"{float(gate['accuracy_interval_95'][1]):.6f}]`"
            ),
            (
                "- Maximum swap-logit error: "
                f"`{float(report['candidate_order']['maximum_observed_swap_logit_error']):.9f}`"
            ),
            "",
            "Future transfer remains blocked unless the source gate passes.",
            "",
        ]
    )


def _source_gate(test_metrics: Mapping[str, Any]) -> dict[str, Any]:
    total = int(test_metrics["records"])
    accuracy = float(test_metrics["accuracy"])
    correct = int(round(accuracy * total))
    lower, upper = _wilson_interval(correct, total)
    prior_log_loss = math.log(2.0)
    return {
        "passed": lower > 0.5 and float(test_metrics["log_loss"]) < prior_log_loss,
        "accuracy_interval_95": [lower, upper],
        "class_prior_accuracy": 0.5,
        "class_prior_log_loss": prior_log_loss,
        "requirements": [
            "test_accuracy_wilson_lower_bound_above_0.5",
            "test_log_loss_below_log_2",
            "candidate_swap_logit_reversal",
        ],
    }


def _encode_mean_pooled(
    *,
    stack: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    maximum_length: int,
    device: Any,
) -> Any:
    torch = stack["torch"]
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=maximum_length,
                return_tensors="pt",
            )
            batch = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = model(**batch, return_dict=True)
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().float().cpu())
    result = torch.cat(vectors, dim=0).contiguous()
    if result.shape[0] != len(texts) or not bool(torch.isfinite(result).all().item()):
        raise ValueError("invalid Step 8 embedding matrix")
    return result


def _context_view(row: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "[CONTEXT_BEFORE]",
            str(row.get("context_before", "")),
            "[CONTEXT_AFTER]",
            str(row.get("context_after", "")),
        )
    )


def _candidate_view(row: Mapping[str, Any], field: str) -> str:
    return "\n".join(
        (
            "[CONTEXT_BEFORE]",
            str(row.get("context_before", "")),
            "[CANDIDATE]",
            str(row[field]),
            "[CONTEXT_AFTER]",
            str(row.get("context_after", "")),
        )
    )


def _validate_episode_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "episode_id",
        "lineage_id",
        "candidate_a",
        "candidate_b",
        "selected_index",
        "future_revised",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for episode in episodes:
        missing = required.difference(episode)
        if missing:
            raise ValueError(f"Step 8 episode missing fields: {sorted(missing)}")
        episode_id = str(episode["episode_id"])
        if episode_id in seen:
            raise ValueError(f"duplicate Step 8 episode: {episode_id}")
        seen.add(episode_id)
        selected_index = episode["selected_index"]
        if type(selected_index) is not int or selected_index not in (0, 1):
            raise ValueError(f"invalid selected_index for {episode_id}")
        rows.append(dict(episode))
    if not rows:
        raise ValueError("Step 8 episode source is empty")
    return rows


def _load_teacher_probabilities(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    values: dict[str, float] = {}
    for record in load_jsonl(path.expanduser().resolve()):
        episode_id = str(record.get("episode_id", ""))
        probability = float(record.get("probability_a"))
        if not episode_id or episode_id in values or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid teacher probability record")
        values[episode_id] = probability
    return values


def _parse_rankers(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return RANKER_NAMES
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = set(parsed).difference(RANKER_NAMES)
    if not parsed or unknown:
        raise ValueError(f"invalid ranker selection: {sorted(unknown)}")
    return tuple(name for name in RANKER_NAMES if name in parsed)


def _load_and_validate_contract(root: Path) -> dict[str, Any]:
    contract = load_json(root / "contract.json")
    expected = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not expected or canonical_json_sha256(payload) != expected:
        raise ValueError("Step 8 contract hash is missing or invalid")
    for source in contract["sources"].values():
        path = Path(str(source["path"]))
        if sha256_file(path) != str(source["sha256"]):
            raise ValueError(f"Step 8 source changed: {path}")
    snapshot = Path(str(contract["embedder"]["snapshot_path"]))
    if sha256_directory(snapshot) != str(contract["embedder"]["snapshot_sha256"]):
        raise ValueError("Step 8 embedder snapshot changed")
    return contract


def _require_ranker_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("Step 8 requires torch and safetensors; install with .[train]") from exc
    return torch, load_file, save_file


def _wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (rate + z2 / (2.0 * total)) / denominator
    spread = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)
