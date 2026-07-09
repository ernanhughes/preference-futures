"""Optional PyTorch/Transformers runtime for Step 3 representation training."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.training.common import (
    CLASSIFICATION_REGIMES,
    LANGUAGE_ADAPTATION_REGIME,
    TRAINING_RUN_SCHEMA_VERSION,
    load_json,
    load_jsonl,
    parse_int_selection,
    parse_regime_selection,
    sha256_directory,
    sha256_file,
    write_json,
    write_jsonl,
)
from preference_futures.training.contract import (
    build_training_contract,
    validate_training_contract,
    write_training_contract,
)
from preference_futures.training.data import (
    ClassificationExample,
    MaskedLanguageExample,
    deterministic_training_batches,
    load_source_store,
    materialize_record,
    sequential_validation_batches,
)


def prepare_training(
    *,
    corpora_directory: Path,
    episodes_path: Path,
    output_directory: Path,
    model_id: str = "distilbert/distilbert-base-uncased",
    model_revision: str = "main",
    seed: int = 17,
    maximum_sequence_length: int = 256,
    batch_size: int = 16,
    update_steps: int = 600,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 60,
    gradient_clip_norm: float = 1.0,
    log_every_steps: int = 25,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve and snapshot one base encoder, then freeze the Step 3 contract."""

    stack = _require_training_stack()
    output = output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"training output is not empty; pass --force to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    resolved_revision = _resolve_revision(
        model_id,
        model_revision,
        model_info=stack["model_info"],
    )
    snapshot = output / "base-snapshot"
    encoder_dir = snapshot / "encoder"
    tokenizer_dir = snapshot / "tokenizer"

    _set_seed(stack["torch"], seed)
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        model_id,
        revision=None if resolved_revision == "local" else resolved_revision,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Step 3 requires a fast tokenizer for deterministic whole-word masks")
    if tokenizer.mask_token_id is None:
        raise ValueError("Step 3 language adaptation requires a tokenizer mask token")
    encoder = stack["AutoModel"].from_pretrained(
        model_id,
        revision=None if resolved_revision == "local" else resolved_revision,
    )
    encoder.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)

    model_metadata = {
        "model_id": model_id,
        "requested_revision": model_revision,
        "resolved_revision": resolved_revision,
        "encoder_class": type(encoder).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": stack["transformers_version"],
        "torch_version_at_prepare": stack["torch"].__version__,
        "huggingface_hub_version": stack["huggingface_hub_version"],
    }
    write_json(output / "model-source.json", model_metadata)
    contract = build_training_contract(
        corpora_directory=corpora_directory,
        episodes_path=episodes_path,
        output_directory=output,
        base_snapshot_directory=snapshot,
        model_metadata=model_metadata,
        seed=seed,
        maximum_sequence_length=maximum_sequence_length,
        batch_size=batch_size,
        update_steps=update_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_clip_norm=gradient_clip_norm,
        log_every_steps=log_every_steps,
    )
    write_training_contract(output, contract)
    del encoder
    gc.collect()
    return contract


def run_training_jobs(
    training_directory: Path,
    *,
    folds: str = "all",
    regimes: str = "all",
    device: str = "auto",
    smoke_steps: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run selected jobs under the frozen contract, resetting the fold seed for every regime."""

    stack = _require_training_stack()
    torch = stack["torch"]
    training = training_directory.expanduser().resolve()
    contract_path = training / "contract.json"
    contract = load_json(contract_path)
    validate_training_contract(contract)
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_regimes = parse_regime_selection(regimes)
    optimisation = contract["optimisation"]
    confirmatory_steps = int(optimisation["update_steps"])
    actual_steps = smoke_steps if smoke_steps is not None else confirmatory_steps
    if actual_steps < 1 or actual_steps > confirmatory_steps:
        raise ValueError("smoke_steps must be between 1 and the frozen update budget")
    non_confirmatory = smoke_steps is not None
    run_root = training / ("smoke-runs" if non_confirmatory else "runs")
    run_root.mkdir(parents=True, exist_ok=True)

    source_store = load_source_store(
        Path(contract["sources"]["episodes"]["path"]),
        Path(contract["sources"]["temporal_pairs"]["path"]),
    )
    snapshot = Path(contract["model"]["base_snapshot_path"])
    encoder_dir = snapshot / "encoder"
    tokenizer_dir = snapshot / "tokenizer"
    tokenizer = stack["AutoTokenizer"].from_pretrained(tokenizer_dir, use_fast=True)
    base_encoder = stack["AutoModel"].from_pretrained(encoder_dir)
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in base_encoder.state_dict().items()}
    del base_encoder
    resolved_device = _resolve_device(torch, device)

    jobs_by_key = {
        (int(job["fold"]), str(job["regime"])): job for job in contract["jobs"]
    }
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fold in selected_folds:
        for regime in selected_regimes:
            job = jobs_by_key[(fold, regime)]
            run_directory = run_root / f"fold-{fold:02d}" / regime
            if not force and _completed_run_matches(
                run_directory,
                contract_sha256=str(contract["contract_sha256"]),
                expected_steps=actual_steps,
                non_confirmatory=non_confirmatory,
            ):
                skipped.append({"fold": fold, "regime": regime})
                continue
            report = _run_one_job(
                stack=stack,
                contract=contract,
                job=job,
                source_store=source_store,
                tokenizer=tokenizer,
                base_state=base_state,
                output_directory=run_directory,
                device=resolved_device,
                update_steps=actual_steps,
                non_confirmatory=non_confirmatory,
                force=force,
            )
            completed.append(report)

    summary = {
        "contract_sha256": contract["contract_sha256"],
        "training_directory": str(training),
        "run_root": str(run_root),
        "non_confirmatory": non_confirmatory,
        "selected_folds": list(selected_folds),
        "selected_regimes": list(selected_regimes),
        "completed_jobs": [
            {"fold": report["fold"], "regime": report["regime"]} for report in completed
        ],
        "skipped_jobs": skipped,
        "device": str(resolved_device),
        "update_steps_per_job": actual_steps,
    }
    write_json(run_root / "last-run-summary.json", summary)
    return summary


def _run_one_job(
    *,
    stack: Mapping[str, Any],
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
    source_store: Any,
    tokenizer: Any,
    base_state: Mapping[str, Any],
    output_directory: Path,
    device: Any,
    update_steps: int,
    non_confirmatory: bool,
    force: bool,
) -> dict[str, Any]:
    torch = stack["torch"]
    fold = int(job["fold"])
    regime = str(job["regime"])
    train_path = Path(job["train"]["path"])
    validation_path = Path(job["validation"]["path"])
    if sha256_file(train_path) != str(job["train"]["sha256"]):
        raise ValueError(f"training corpus changed: fold {fold} {regime}")
    if sha256_file(validation_path) != str(job["validation"]["sha256"]):
        raise ValueError(f"validation corpus changed: fold {fold} {regime}")
    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    if len(train_records) != int(job["train"]["records"]):
        raise ValueError(f"training record count changed: fold {fold} {regime}")
    if len(validation_records) != int(job["validation"]["records"]):
        raise ValueError(f"validation record count changed: fold {fold} {regime}")

    optimisation = contract["optimisation"]
    batch_size = int(optimisation["batch_size"])
    max_length = int(optimisation["maximum_sequence_length"])
    fold_seed = int(contract["seed"]) + fold * 1000
    _set_seed(torch, fold_seed)
    model = _instantiate_task_model(stack, regime, base_state, contract)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimisation["learning_rate"]),
        weight_decay=float(optimisation["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _linear_schedule(
            warmup_steps=int(optimisation["warmup_steps"]),
            total_steps=int(optimisation["update_steps"]),
        ),
    )
    batches = deterministic_training_batches(
        len(train_records),
        batch_size=batch_size,
        update_steps=update_steps,
        seed=fold_seed,
    )
    metrics: list[dict[str, Any]] = []
    running_loss = 0.0
    fallback_masks = 0
    optimizer.zero_grad(set_to_none=True)
    log_every = int(optimisation["log_every_steps"])
    for step, indices in enumerate(batches, start=1):
        records = [train_records[index] for index in indices]
        examples = [materialize_record(record, source_store) for record in records]
        if regime == LANGUAGE_ADAPTATION_REGIME:
            batch, fallback_count = _collate_mlm(
                stack,
                tokenizer,
                examples,
                max_length=max_length,
                device=device,
            )
            fallback_masks += fallback_count
        else:
            batch = _collate_classification(
                stack,
                tokenizer,
                examples,
                max_length=max_length,
                device=device,
            )
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at fold {fold} {regime} step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(optimisation["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach().cpu())
        if step % log_every == 0 or step == update_steps:
            metrics.append(
                {
                    "step": step,
                    "mean_training_loss_since_last_log": running_loss
                    / (log_every if step % log_every == 0 else step % log_every),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "examples_seen": step * batch_size,
                    "padded_token_positions": step * batch_size * max_length,
                }
            )
            running_loss = 0.0

    validation = _evaluate(
        stack=stack,
        model=model,
        tokenizer=tokenizer,
        records=validation_records,
        source_store=source_store,
        regime=regime,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_directory.exists():
        if not force:
            raise ValueError(f"run output already exists: {output_directory}")
        shutil.rmtree(output_directory)
    temporary.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temporary / "task-model", safe_serialization=True)
    model.base_model.save_pretrained(temporary / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "tokenizer")
    write_jsonl(temporary / "metrics.jsonl", metrics)

    report = {
        "training_run_schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "regime": regime,
        "objective": job["objective"],
        "non_confirmatory": non_confirmatory,
        "seed": fold_seed,
        "source_files": {
            "train_sha256": job["train"]["sha256"],
            "validation_sha256": job["validation"]["sha256"],
        },
        "model": {
            "model_id": contract["model"]["model_id"],
            "resolved_revision": contract["model"]["resolved_revision"],
            "initial_encoder_snapshot_sha256": contract["model"][
                "base_snapshot_sha256"
            ],
            "task_model_class": type(model).__name__,
            "encoder_class": type(model.base_model).__name__,
        },
        "optimisation": {
            "optimizer_steps_completed": update_steps,
            "batch_size": batch_size,
            "maximum_sequence_length": max_length,
            "padded_token_positions": update_steps * batch_size * max_length,
            "examples_seen": update_steps * batch_size,
            "precision": "fp32",
            "checkpoint_step": update_steps,
            "early_stopping_used": False,
        },
        "validation": validation,
        "mask_fallback_examples": fallback_masks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": stack["transformers_version"],
            "device": str(device),
            "device_name": _device_name(torch, device),
            "cuda_version": getattr(torch.version, "cuda", None),
        },
        "artifacts": {
            "task_model_sha256": sha256_directory(temporary / "task-model"),
            "encoder_sha256": sha256_directory(temporary / "encoder"),
            "tokenizer_sha256": sha256_directory(temporary / "tokenizer"),
        },
    }
    write_json(temporary / "run.json", report)
    temporary.replace(output_directory)

    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _instantiate_task_model(
    stack: Mapping[str, Any],
    regime: str,
    base_state: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Any:
    snapshot = Path(contract["model"]["base_snapshot_path"])
    config = stack["AutoConfig"].from_pretrained(snapshot / "encoder")
    if regime == LANGUAGE_ADAPTATION_REGIME:
        model = stack["AutoModelForMaskedLM"].from_config(config)
    elif regime in CLASSIFICATION_REGIMES:
        config.num_labels = 2
        model = stack["AutoModelForSequenceClassification"].from_config(config)
    else:
        raise ValueError(f"unsupported training regime: {regime}")
    model.base_model.load_state_dict(base_state, strict=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def _collate_classification(
    stack: Mapping[str, Any],
    tokenizer: Any,
    examples: Sequence[ClassificationExample | MaskedLanguageExample],
    *,
    max_length: int,
    device: Any,
) -> dict[str, Any]:
    if not all(isinstance(example, ClassificationExample) for example in examples):
        raise ValueError("classification batch contains a non-classification example")
    texts = [example.text for example in examples if isinstance(example, ClassificationExample)]
    targets = [example.target for example in examples if isinstance(example, ClassificationExample)]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {name: tensor.to(device) for name, tensor in encoded.items()}
    batch["labels"] = stack["torch"].tensor(targets, dtype=stack["torch"].long, device=device)
    return batch


def _collate_mlm(
    stack: Mapping[str, Any],
    tokenizer: Any,
    examples: Sequence[ClassificationExample | MaskedLanguageExample],
    *,
    max_length: int,
    device: Any,
) -> tuple[dict[str, Any], int]:
    torch = stack["torch"]
    if not all(isinstance(example, MaskedLanguageExample) for example in examples):
        raise ValueError("MLM batch contains a non-language example")
    words = [list(example.words) for example in examples if isinstance(example, MaskedLanguageExample)]
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].clone()
    labels = torch.full_like(input_ids, -100)
    fallback_count = 0
    for batch_index, example in enumerate(examples):
        assert isinstance(example, MaskedLanguageExample)
        word_ids = encoded.word_ids(batch_index=batch_index)
        selected = set(example.mask_word_indices)
        positions = [
            index
            for index, word_id in enumerate(word_ids)
            if word_id is not None and word_id in selected
        ]
        if not positions:
            positions = [
                index for index, word_id in enumerate(word_ids) if word_id is not None
            ][:1]
            fallback_count += 1
        for position in positions:
            labels[batch_index, position] = input_ids[batch_index, position]
            input_ids[batch_index, position] = tokenizer.mask_token_id
    batch = {name: tensor.to(device) for name, tensor in encoded.items()}
    batch["input_ids"] = input_ids.to(device)
    batch["labels"] = labels.to(device)
    return batch, fallback_count


def _evaluate(
    *,
    stack: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    source_store: Any,
    regime: str,
    batch_size: int,
    max_length: int,
    device: Any,
) -> dict[str, Any]:
    torch = stack["torch"]
    model.eval()
    loss_weighted = 0.0
    weight = 0
    correct = 0
    fallback_masks = 0
    with torch.no_grad():
        for indices in sequential_validation_batches(len(records), batch_size=batch_size):
            examples = [materialize_record(records[index], source_store) for index in indices]
            if regime == LANGUAGE_ADAPTATION_REGIME:
                batch, fallback_count = _collate_mlm(
                    stack,
                    tokenizer,
                    examples,
                    max_length=max_length,
                    device=device,
                )
                fallback_masks += fallback_count
                outputs = model(**batch)
                labels = batch["labels"]
                mask = labels.ne(-100)
                supervised = int(mask.sum().item())
                predictions = outputs.logits.argmax(dim=-1)
                correct += int((predictions[mask] == labels[mask]).sum().item())
                loss_weighted += float(outputs.loss.detach().cpu()) * supervised
                weight += supervised
            else:
                batch = _collate_classification(
                    stack,
                    tokenizer,
                    examples,
                    max_length=max_length,
                    device=device,
                )
                outputs = model(**batch)
                labels = batch["labels"]
                predictions = outputs.logits.argmax(dim=-1)
                batch_count = int(labels.numel())
                correct += int((predictions == labels).sum().item())
                loss_weighted += float(outputs.loss.detach().cpu()) * batch_count
                weight += batch_count
    model.train()
    mean_loss = loss_weighted / max(1, weight)
    result = {
        "records": len(records),
        "supervised_units": weight,
        "mean_loss": mean_loss,
        "accuracy": correct / max(1, weight),
        "mask_fallback_examples": fallback_masks,
    }
    if regime == LANGUAGE_ADAPTATION_REGIME:
        result["perplexity"] = math.exp(min(20.0, mean_loss))
    return result


def _linear_schedule(*, warmup_steps: int, total_steps: int) -> Any:
    def schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / max(1, warmup_steps)
        remaining = total_steps - current_step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining) / decay_steps)

    return schedule


def _completed_run_matches(
    run_directory: Path,
    *,
    contract_sha256: str,
    expected_steps: int,
    non_confirmatory: bool,
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
            and report.get("non_confirmatory") is non_confirmatory
            and report.get("optimisation", {}).get("optimizer_steps_completed")
            == expected_steps
            and sha256_directory(run_directory / "task-model")
            == artifacts["task_model_sha256"]
            and sha256_directory(run_directory / "encoder") == artifacts["encoder_sha256"]
            and sha256_directory(run_directory / "tokenizer")
            == artifacts["tokenizer_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _resolve_revision(model_id: str, requested_revision: str, *, model_info: Any) -> str:
    if Path(model_id).expanduser().exists():
        return "local"
    info = model_info(model_id, revision=requested_revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(f"could not resolve immutable model revision: {model_id}")
    return resolved


def _resolve_device(torch: Any, requested: str) -> Any:
    value = requested.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            value = "mps"
        else:
            value = "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _set_seed(torch: Any, seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device_name(torch: Any, device: Any) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return device.type


def _require_training_stack() -> dict[str, Any]:
    try:
        import torch
        import transformers
        from huggingface_hub import __version__ as huggingface_hub_version
        from huggingface_hub import model_info
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Step 3 training dependencies are missing. Install with: "
            "python -m pip install -e '.[train]'"
        ) from exc
    return {
        "torch": torch,
        "transformers_version": transformers.__version__,
        "huggingface_hub_version": huggingface_hub_version,
        "model_info": model_info,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
    }
