"""Optional model runtime for Step 5 frozen representation extraction."""

from __future__ import annotations

import gc
import hashlib
import platform
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.representations.common import (
    PARTITIONS,
    REPRESENTATION_RUN_SCHEMA_VERSION,
    parse_arm_selection,
)
from preference_futures.representations.contract import (
    build_representation_contract,
    validate_representation_contract,
    write_representation_contract,
)
from preference_futures.training.common import (
    load_json,
    parse_int_selection,
    sha256_file,
    write_json,
    write_jsonl,
)
from preference_futures.training.data import load_source_store, serialise_episode
from preference_futures.training.runtime import (
    _device_name,
    _require_training_stack,
    _resolve_device,
    _set_seed,
)


def prepare_representations(
    *,
    selection_manifest_path: Path,
    training_directory: Path,
    output_directory: Path,
    batch_size: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze Step 5 before any representation matrix is written."""

    output = output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(
                f"representation output is not empty; pass --force to replace it: {output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = build_representation_contract(
        selection_manifest_path=selection_manifest_path,
        training_directory=training_directory,
        output_directory=output,
        batch_size=batch_size,
    )
    write_representation_contract(output, contract)
    return contract


def run_representation_jobs(
    representation_directory: Path,
    *,
    folds: str = "all",
    arms: str = "all",
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Extract selected frozen encoders under the committed Step 5 contract."""

    stack = _require_training_stack()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "Step 5 requires safetensors. Install with: python -m pip install -e '.[train]'"
        ) from exc

    torch = stack["torch"]
    root = representation_directory.expanduser().resolve()
    contract = load_json(root / "contract.json")
    validate_representation_contract(contract)
    selected_folds = parse_int_selection(folds, upper_bound=int(contract["outer_folds"]))
    selected_arms = parse_arm_selection(arms)
    selected_keys = {(fold, arm) for fold in selected_folds for arm in selected_arms}
    jobs = [
        job
        for job in contract["jobs"]
        if (int(job["fold"]), str(job["regime"])) in selected_keys
    ]
    if len(jobs) != len(selected_keys):
        raise ValueError("Step 5 selected jobs do not match the frozen contract")

    source_store = load_source_store(
        Path(contract["sources"]["episodes"]["path"]),
        Path(contract["sources"]["temporal_pairs"]["path"]),
    )
    split_manifest = load_json(Path(contract["sources"]["split_manifest"]["path"]))
    episode_ids = sorted(source_store.episodes)
    episodes = [source_store.episodes[episode_id] for episode_id in episode_ids]
    texts = [serialise_episode(episode) for episode in episodes]
    input_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    partition_indices = _partition_indices(
        episodes,
        split_manifest,
        outer_folds=int(contract["outer_folds"]),
    )

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        Path(contract["tokenizer"]["path"]),
        use_fast=True,
    )
    resolved_device = _resolve_device(torch, device)
    _set_seed(torch, int(contract["seed"]))
    run_root = root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    hash_counts = Counter(str(job["encoder_sha256"]) for job in jobs)
    reusable_embeddings: dict[str, Any] = {}
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for job in sorted(jobs, key=lambda item: (int(item["fold"]), str(item["regime"]))):
        fold = int(job["fold"])
        regime = str(job["regime"])
        run_directory = run_root / f"fold-{fold:02d}" / regime
        if not force and _completed_run_matches(
            run_directory,
            contract_sha256=str(contract["contract_sha256"]),
            encoder_sha256=str(job["encoder_sha256"]),
        ):
            skipped.append({"fold": fold, "regime": regime})
            continue

        encoder_sha256 = str(job["encoder_sha256"])
        print(f"Extracting fold {fold:02d} / {regime} ...", flush=True)
        if encoder_sha256 in reusable_embeddings:
            full_representations = reusable_embeddings[encoder_sha256]
            model_class = "cached-shared-encoder"
        else:
            model = stack["AutoModel"].from_pretrained(Path(job["encoder_path"]))
            model.to(resolved_device)
            model.eval()
            model_class = type(model).__name__
            full_representations = _encode_texts(
                stack=stack,
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                batch_size=int(contract["representation"]["batch_size"]),
                maximum_length=int(
                    contract["representation"]["maximum_sequence_length"]
                ),
                device=resolved_device,
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hash_counts[encoder_sha256] > 1:
                reusable_embeddings[encoder_sha256] = full_representations

        report = _persist_representation_run(
            save_file=save_file,
            torch=torch,
            contract=contract,
            job=job,
            output_directory=run_directory,
            full_representations=full_representations,
            partition_indices=partition_indices[fold],
            episodes=episodes,
            input_hashes=input_hashes,
            device=resolved_device,
            device_name=_device_name(torch, resolved_device),
            model_class=model_class,
            transformers_version=stack["transformers_version"],
            force=force,
        )
        completed.append(
            {
                "fold": fold,
                "regime": regime,
                "hidden_size": report["representation"]["hidden_size"],
            }
        )
        if hash_counts[encoder_sha256] == 1:
            del full_representations
            gc.collect()

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


def _partition_indices(
    episodes: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    outer_folds: int,
) -> dict[int, dict[str, list[int]]]:
    assignments = split_manifest.get("lineage_to_outer_fold")
    if not isinstance(assignments, Mapping):
        raise ValueError("Step 1 split manifest has no lineage assignments")
    episode_lineages = {str(episode["lineage_id"]) for episode in episodes}
    if set(map(str, assignments)) != episode_lineages:
        raise ValueError("Step 1 lineage assignments do not match Step 5 episodes")
    result: dict[int, dict[str, list[int]]] = {}
    for fold in range(outer_folds):
        validation_bucket = (fold + 1) % outer_folds
        partitions = {name: [] for name in PARTITIONS}
        for index, episode in enumerate(episodes):
            bucket = int(assignments[str(episode["lineage_id"])])
            if bucket == fold:
                partition = "test"
            elif bucket == validation_bucket:
                partition = "validation"
            else:
                partition = "train"
            partitions[partition].append(index)
        if any(not partitions[name] for name in PARTITIONS):
            raise ValueError(f"fold {fold} contains an empty Step 5 partition")
        result[fold] = partitions
    return result


def _encode_texts(
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
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                list(batch_texts),
                padding="max_length",
                truncation=True,
                max_length=maximum_length,
                return_tensors="pt",
            )
            batch = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = model(**batch, return_dict=True)
            cls_vectors = outputs.last_hidden_state[:, 0, :].detach().float().cpu()
            vectors.append(cls_vectors)
    result = torch.cat(vectors, dim=0).contiguous()
    if result.ndim != 2 or result.shape[0] != len(texts):
        raise ValueError("Step 5 encoder returned an invalid representation matrix")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("Step 5 encoder returned non-finite representations")
    return result


def _persist_representation_run(
    *,
    save_file: Any,
    torch: Any,
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
    output_directory: Path,
    full_representations: Any,
    partition_indices: Mapping[str, Sequence[int]],
    episodes: Sequence[Mapping[str, Any]],
    input_hashes: Sequence[str],
    device: Any,
    device_name: str,
    model_class: str,
    transformers_version: str,
    force: bool,
) -> dict[str, Any]:
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_directory.exists():
        if not force:
            raise ValueError(f"Step 5 run output already exists: {output_directory}")
        shutil.rmtree(output_directory)
    temporary.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {}
    partition_counts: dict[str, int] = {}
    hidden_size = int(full_representations.shape[1])
    for partition in PARTITIONS:
        indices = list(partition_indices[partition])
        index_tensor = torch.tensor(indices, dtype=torch.long)
        matrix = full_representations.index_select(0, index_tensor).contiguous()
        vector_path = temporary / f"{partition}.safetensors"
        rows_path = temporary / f"{partition}.rows.jsonl"
        save_file(
            {"representations": matrix},
            str(vector_path),
            metadata={
                "contract_sha256": str(contract["contract_sha256"]),
                "fold": str(job["fold"]),
                "regime": str(job["regime"]),
                "partition": partition,
                "pooling": str(contract["representation"]["pooling"]),
                "dtype": "float32",
            },
        )
        rows = [
            {
                "row_index": row_index,
                "episode_id": str(episodes[source_index]["episode_id"]),
                "lineage_id": str(episodes[source_index]["lineage_id"]),
                "input_sha256": input_hashes[source_index],
            }
            for row_index, source_index in enumerate(indices)
        ]
        write_jsonl(rows_path, rows)
        partition_counts[partition] = len(indices)
        artifacts[partition] = {
            "representations_path": f"{partition}.safetensors",
            "representations_sha256": sha256_file(vector_path),
            "rows_path": f"{partition}.rows.jsonl",
            "rows_sha256": sha256_file(rows_path),
            "rows": len(indices),
            "shape": [len(indices), hidden_size],
            "dtype": "float32",
        }

    report = {
        "representation_run_schema_version": REPRESENTATION_RUN_SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "fold": int(job["fold"]),
        "regime": str(job["regime"]),
        "arm_kind": str(job["arm_kind"]),
        "encoder_path": str(job["encoder_path"]),
        "encoder_sha256": str(job["encoder_sha256"]),
        "source_task_status": job.get("source_task_status"),
        "representation": {
            "input_view": contract["representation"]["input_view"],
            "pooling": contract["representation"]["pooling"],
            "hidden_size": hidden_size,
            "dtype": "float32",
            "partition_counts": partition_counts,
            "future_fields_exposed": False,
            "selected_index_exposed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": str(device),
            "device_name": device_name,
            "cuda_version": getattr(torch.version, "cuda", None),
            "model_class": model_class,
        },
        "artifacts": artifacts,
    }
    write_json(temporary / "run.json", report)
    temporary.replace(output_directory)
    return report


def _completed_run_matches(
    run_directory: Path,
    *,
    contract_sha256: str,
    encoder_sha256: str,
) -> bool:
    report_path = run_directory / "run.json"
    if not report_path.exists():
        return False
    try:
        report = load_json(report_path)
        if (
            report.get("status") != "complete"
            or report.get("contract_sha256") != contract_sha256
            or report.get("encoder_sha256") != encoder_sha256
        ):
            return False
        for partition in PARTITIONS:
            artifact = report["artifacts"][partition]
            if sha256_file(run_directory / f"{partition}.safetensors") != artifact[
                "representations_sha256"
            ]:
                return False
            if sha256_file(run_directory / f"{partition}.rows.jsonl") != artifact[
                "rows_sha256"
            ]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False
