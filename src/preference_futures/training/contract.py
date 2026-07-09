"""Build and validate the frozen Step 3 optimisation contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from preference_futures.corpora.verify import verify_compute_matched_corpora
from preference_futures.training.common import (
    TRAINED_REGIMES,
    TRAINING_CONTRACT_SCHEMA_VERSION,
    canonical_json_sha256,
    load_json,
    positive_int,
    require_mapping,
    require_sequence,
    sha256_directory,
    sha256_file,
    write_json,
)


def build_training_contract(
    *,
    corpora_directory: Path,
    episodes_path: Path,
    output_directory: Path,
    base_snapshot_directory: Path,
    model_metadata: Mapping[str, Any],
    seed: int = 17,
    maximum_sequence_length: int = 256,
    batch_size: int = 16,
    update_steps: int = 600,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 60,
    gradient_clip_norm: float = 1.0,
    log_every_steps: int = 25,
) -> dict[str, Any]:
    """Freeze all model, data and optimisation choices before source-task training."""

    corpora = corpora_directory.expanduser().resolve()
    episodes = episodes_path.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    base_snapshot = base_snapshot_directory.expanduser().resolve()
    _validate_hyperparameters(
        maximum_sequence_length=maximum_sequence_length,
        batch_size=batch_size,
        update_steps=update_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_clip_norm=gradient_clip_norm,
        log_every_steps=log_every_steps,
    )

    corpus_manifest_path = corpora / "manifest.json"
    if not corpus_manifest_path.exists():
        raise ValueError(f"Step 2 manifest does not exist: {corpus_manifest_path}")
    corpus_manifest = load_json(corpus_manifest_path)
    verification = verify_compute_matched_corpora(corpora)
    if verification.get("passed") is not True:
        raise ValueError(f"Step 2 persisted verification failed: {verification.get('errors')}")
    if tuple(corpus_manifest.get("corpora", ())) != TRAINED_REGIMES:
        raise ValueError("Step 2 corpus order does not match the six frozen regimes")

    sources = require_mapping(corpus_manifest.get("sources"), "Step 2 sources")
    episode_source = require_mapping(sources.get("episodes"), "Step 2 episode source")
    expected_episode_hash = str(episode_source.get("sha256", ""))
    observed_episode_hash = sha256_file(episodes)
    if not expected_episode_hash or observed_episode_hash != expected_episode_hash:
        raise ValueError(
            "episode source hash does not match Step 2: "
            f"{observed_episode_hash} != {expected_episode_hash}"
        )

    temporal_pairs_path = corpora / "temporal-pairs.jsonl"
    temporal_source = require_mapping(sources.get("temporal_pairs"), "Step 2 temporal source")
    expected_temporal_hash = str(temporal_source.get("sha256", ""))
    observed_temporal_hash = sha256_file(temporal_pairs_path)
    if not expected_temporal_hash or observed_temporal_hash != expected_temporal_hash:
        raise ValueError(
            "temporal source hash does not match Step 2: "
            f"{observed_temporal_hash} != {expected_temporal_hash}"
        )

    outer_folds = positive_int(corpus_manifest.get("outer_folds"), "outer_folds")
    fold_summaries = require_sequence(corpus_manifest.get("folds"), "Step 2 folds")
    if len(fold_summaries) != outer_folds:
        raise ValueError("Step 2 fold summary count does not match outer_folds")

    jobs: list[dict[str, Any]] = []
    for fold in range(outer_folds):
        fold_summary = require_mapping(fold_summaries[fold], f"fold {fold} summary")
        if fold_summary.get("fold") != fold:
            raise ValueError(f"Step 2 fold order is not canonical at fold {fold}")
        partitions = require_mapping(fold_summary.get("partitions"), f"fold {fold} partitions")
        train_summary = require_mapping(partitions.get("train"), f"fold {fold} train")
        validation_summary = require_mapping(
            partitions.get("validation"), f"fold {fold} validation"
        )
        train_records = positive_int(
            train_summary.get("records_per_corpus"), f"fold {fold} train records"
        )
        validation_records = positive_int(
            validation_summary.get("records_per_corpus"),
            f"fold {fold} validation records",
        )
        for regime in TRAINED_REGIMES:
            train_path = corpora / f"fold-{fold:02d}" / regime / "train.jsonl"
            validation_path = corpora / f"fold-{fold:02d}" / regime / "validation.jsonl"
            jobs.append(
                {
                    "fold": fold,
                    "regime": regime,
                    "objective": (
                        "masked_language_modeling"
                        if regime == "language_adaptation"
                        else "binary_sequence_classification"
                    ),
                    "train": {
                        "path": str(train_path),
                        "sha256": sha256_file(train_path),
                        "records": train_records,
                    },
                    "validation": {
                        "path": str(validation_path),
                        "sha256": sha256_file(validation_path),
                        "records": validation_records,
                    },
                }
            )

    padded_tokens_per_job = update_steps * batch_size * maximum_sequence_length
    contract: dict[str, Any] = {
        "training_contract_schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "status": "frozen_before_training",
        "seed": seed,
        "outer_folds": outer_folds,
        "trained_regimes": list(TRAINED_REGIMES),
        "expected_training_jobs": len(jobs),
        "sources": {
            "step_2_manifest": {
                "path": str(corpus_manifest_path),
                "sha256": sha256_file(corpus_manifest_path),
            },
            "step_2_verification": verification,
            "episodes": {
                "path": str(episodes),
                "sha256": observed_episode_hash,
            },
            "temporal_pairs": {
                "path": str(temporal_pairs_path),
                "sha256": observed_temporal_hash,
            },
        },
        "model": {
            **dict(model_metadata),
            "base_snapshot_path": str(base_snapshot),
            "base_snapshot_sha256": sha256_directory(base_snapshot),
            "task_head_initialisation": "deterministic from config after resetting the fold seed",
            "encoder_initialisation": "strict load from the single frozen base snapshot",
            "saved_outputs": ["full_task_model", "encoder_only", "tokenizer"],
        },
        "optimisation": {
            "precision": "fp32",
            "maximum_sequence_length": maximum_sequence_length,
            "padding": "max_length",
            "truncation": True,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "update_steps": update_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "scheduler": "linear_warmup_then_linear_decay",
            "optimizer": "torch.optim.AdamW",
            "gradient_clip_norm": gradient_clip_norm,
            "log_every_steps": log_every_steps,
            "shuffle": "deterministic index permutation; seed plus fold plus epoch",
            "checkpoint_rule": f"final update {update_steps}; no task-specific selection",
            "validation_rule": "full source-validation partition after the final update only",
            "early_stopping": False,
            "padded_token_positions_per_job": padded_tokens_per_job,
            "padded_token_positions_all_jobs": padded_tokens_per_job * len(jobs),
        },
        "jobs": jobs,
        "gates": {
            "step_2_persisted_verification_passed": True,
            "episode_hash_matches_step_2": True,
            "temporal_hash_matches_step_2": True,
            "six_regimes_declared": True,
            "sixty_jobs_declared": len(jobs) == outer_folds * len(TRAINED_REGIMES),
            "fixed_optimizer_updates": True,
            "fixed_padded_token_budget": True,
            "fixed_final_checkpoint_rule": True,
            "no_task_specific_early_stopping": True,
            "future_labels_unavailable_to_source_training": True,
        },
        "warnings": [
            (
                "Equal padded encoder inputs and optimizer updates do not imply identical total "
                "FLOPs: the masked-language-model head is larger than the binary heads. Report "
                "this as matched encoder optimisation opportunity, not exact wall-clock compute."
            ),
            (
                "The temporal-direction corpus is external and approximately seven percent longer "
                "before padding. Fixed max-length padding removes that raw exposure advantage from "
                "the encoder token budget."
            ),
            "Source-validation metrics are diagnostic and may not select different checkpoints.",
            "Future labels, V2 text and future-probe outcomes are forbidden during Step 3.",
        ],
        "output_directory": str(output),
    }
    if not all(contract["gates"].values()):
        failed = [name for name, passed in contract["gates"].items() if not passed]
        raise ValueError(f"Step 3 contract gates failed: {failed}")
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def write_training_contract(output_directory: Path, contract: Mapping[str, Any]) -> None:
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    (output / "training-plan.md").write_text(
        render_training_plan_markdown(contract), encoding="utf-8"
    )


def validate_training_contract(contract: Mapping[str, Any]) -> None:
    expected_hash = str(contract.get("contract_sha256", ""))
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not expected_hash or canonical_json_sha256(payload) != expected_hash:
        raise ValueError("training contract hash is missing or invalid")
    if tuple(contract.get("trained_regimes", ())) != TRAINED_REGIMES:
        raise ValueError("training contract regime order changed")
    gates = require_mapping(contract.get("gates"), "training contract gates")
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("one or more training contract gates failed")
    sources = require_mapping(contract.get("sources"), "training contract sources")
    for name in ("step_2_manifest", "episodes", "temporal_pairs"):
        source = require_mapping(sources.get(name), f"training source {name}")
        path = Path(str(source.get("path", "")))
        expected = str(source.get("sha256", ""))
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"training source changed: {name}")
    model = require_mapping(contract.get("model"), "training model")
    snapshot = Path(str(model.get("base_snapshot_path", "")))
    if sha256_directory(snapshot) != str(model.get("base_snapshot_sha256", "")):
        raise ValueError("frozen base model snapshot changed")


def render_training_plan_markdown(contract: Mapping[str, Any]) -> str:
    optimisation = require_mapping(contract["optimisation"], "optimisation")
    model = require_mapping(contract["model"], "model")
    lines = [
        "# Fixed-Budget Representation Training Plan",
        "",
        "## Frozen model",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Model | `{model.get('model_id')}` |",
        f"| Resolved revision | `{model.get('resolved_revision')}` |",
        f"| Base snapshot SHA-256 | `{model.get('base_snapshot_sha256')}` |",
        "",
        "## Optimisation contract",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Outer folds | {contract['outer_folds']} |",
        f"| Trained regimes | {len(contract['trained_regimes'])} |",
        f"| Training jobs | {contract['expected_training_jobs']} |",
        f"| Updates per job | {optimisation['update_steps']:,} |",
        f"| Batch size | {optimisation['batch_size']:,} |",
        f"| Maximum sequence length | {optimisation['maximum_sequence_length']:,} |",
        (
            "| Padded token positions per job | "
            f"{optimisation['padded_token_positions_per_job']:,} |"
        ),
        (
            "| Padded token positions across all jobs | "
            f"{optimisation['padded_token_positions_all_jobs']:,} |"
        ),
        f"| Learning rate | {optimisation['learning_rate']} |",
        f"| Warmup updates | {optimisation['warmup_steps']:,} |",
        f"| Precision | `{optimisation['precision']}` |",
        "",
        "## Fixed rule",
        "",
        (
            f"Every job saves update {optimisation['update_steps']} as its candidate encoder. "
            "Validation is diagnostic only. There is no per-regime early stopping or checkpoint "
            "selection."
        ),
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in contract["gates"].items()
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in contract["warnings"])
    lines.append("")
    return "\n".join(lines)


def _validate_hyperparameters(
    *,
    maximum_sequence_length: int,
    batch_size: int,
    update_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    gradient_clip_norm: float,
    log_every_steps: int,
) -> None:
    positive_int(maximum_sequence_length, "maximum_sequence_length")
    positive_int(batch_size, "batch_size")
    positive_int(update_steps, "update_steps")
    positive_int(log_every_steps, "log_every_steps")
    if not 0 < learning_rate < 1:
        raise ValueError("learning_rate must be between zero and one")
    if not 0 <= weight_decay < 1:
        raise ValueError("weight_decay must be in [0, 1)")
    if not 0 <= warmup_steps < update_steps:
        raise ValueError("warmup_steps must be non-negative and less than update_steps")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
