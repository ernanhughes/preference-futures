from __future__ import annotations

from pathlib import Path

import pytest

from preference_futures.representations import contract as contract_module
from preference_futures.representations.common import parse_arm_selection
from preference_futures.representations.runtime import _partition_indices
from preference_futures.selection.diagnostics import ALL_ARMS
from preference_futures.training.common import (
    canonical_json_sha256,
    sha256_directory,
    sha256_file,
    write_json,
)


def _write_encoder(path: Path, payload: bytes) -> str:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(payload)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    return sha256_directory(path)


def test_builds_seven_arm_representation_contract(tmp_path: Path, monkeypatch) -> None:
    training = tmp_path / "training"
    training.mkdir()
    snapshot = training / "base-snapshot"
    tokenizer = snapshot / "tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("{}\n", encoding="utf-8")
    temporal = tmp_path / "temporal.jsonl"
    temporal.write_text("{}\n", encoding="utf-8")
    split = tmp_path / "split.json"
    write_json(split, {"outer_folds": 2, "lineage_to_outer_fold": {"a": 0, "b": 1}})
    step_2 = tmp_path / "step-2.json"
    write_json(
        step_2,
        {
            "sources": {
                "split_manifest": {"path": str(split), "sha256": sha256_file(split)}
            }
        },
    )
    training_contract = {
        "contract_sha256": "step-3-contract",
        "seed": 17,
        "outer_folds": 2,
        "sources": {
            "step_2_manifest": {"path": str(step_2), "sha256": sha256_file(step_2)},
            "episodes": {"path": str(episodes), "sha256": sha256_file(episodes)},
            "temporal_pairs": {"path": str(temporal), "sha256": sha256_file(temporal)},
        },
        "model": {
            "base_snapshot_path": str(snapshot),
            "tokenizer_class": "FixtureTokenizer",
        },
        "optimisation": {"maximum_sequence_length": 256},
    }
    write_json(training / "contract.json", training_contract)
    monkeypatch.setattr(contract_module, "validate_training_contract", lambda value: None)

    entries = []
    base_encoder = snapshot / "encoder"
    base_hash = _write_encoder(base_encoder, b"base")
    for fold in range(2):
        for arm in ALL_ARMS:
            if arm == "generic":
                encoder = base_encoder
                digest = base_hash
                arm_kind = "untouched_base"
            else:
                encoder = tmp_path / "encoders" / f"fold-{fold:02d}" / arm
                digest = _write_encoder(encoder, f"{fold}:{arm}".encode())
                arm_kind = "trained"
            entries.append(
                {
                    "fold": fold,
                    "regime": arm,
                    "arm_kind": arm_kind,
                    "encoder_path": str(encoder),
                    "encoder_sha256": digest,
                    "artifact_valid": True,
                    "eligible_for_downstream": True,
                    "source_task_status": "not_trained" if arm == "generic" else "fixture",
                }
            )
    selection = {
        "encoder_selection_schema_version": 1,
        "status": "frozen_for_step_5",
        "contract_sha256": "step-3-contract",
        "entries": entries,
    }
    selection["manifest_sha256"] = canonical_json_sha256(selection)
    selection_path = tmp_path / "accepted-encoders.json"
    write_json(selection_path, selection)

    contract = contract_module.build_representation_contract(
        selection_manifest_path=selection_path,
        training_directory=training,
        output_directory=tmp_path / "representations",
        batch_size=16,
    )

    assert contract["expected_extraction_jobs"] == 14
    assert contract["expected_partition_artifacts"] == 42
    assert contract["representation"]["pooling"] == "final_hidden_state_first_token"
    assert contract["representation"]["future_label_exposed_to_encoder"] is False
    assert contract["representation"]["selected_index_exposed_to_encoder"] is False
    assert contract["representation"]["batch_size"] == 16
    assert len(contract["jobs"]) == 14


def test_partition_indices_follow_frozen_outer_buckets() -> None:
    episodes = [
        {"episode_id": "a-1", "lineage_id": "a"},
        {"episode_id": "b-1", "lineage_id": "b"},
        {"episode_id": "c-1", "lineage_id": "c"},
    ]
    split = {"lineage_to_outer_fold": {"a": 0, "b": 1, "c": 2}}

    result = _partition_indices(episodes, split, outer_folds=3)

    assert result[0] == {"train": [2], "validation": [1], "test": [0]}
    assert result[1] == {"train": [0], "validation": [2], "test": [1]}
    assert result[2] == {"train": [1], "validation": [0], "test": [2]}


def test_partition_indices_reject_unknown_lineage() -> None:
    episodes = [{"episode_id": "a-1", "lineage_id": "a"}]
    with pytest.raises(ValueError, match="do not match"):
        _partition_indices(
            episodes,
            {"lineage_to_outer_fold": {"other": 0}},
            outer_folds=3,
        )


def test_arm_selection_is_frozen() -> None:
    assert parse_arm_selection("generic,authentic_preference") == (
        "generic",
        "authentic_preference",
    )
    with pytest.raises(ValueError, match="unknown"):
        parse_arm_selection("invented")
