from __future__ import annotations

import json
from pathlib import Path

from preference_futures.splits import (
    build_grouped_split_manifest,
    write_grouped_split_artifacts,
)


def _fixture_records() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    flags: dict[str, dict[str, object]] = {}
    for lineage_index in range(80):
        lineage_id = f"nyt::{lineage_index:03d}"
        episode_count = 1 + lineage_index % 5
        for episode_index in range(episode_count):
            episode_id = f"{lineage_id}::{episode_index}"
            future_revised = (lineage_index + episode_index) % 4 == 0
            selected_index = (lineage_index + episode_index) % 2
            number_changed = (lineage_index + episode_index) % 7 == 0
            number_dominant = number_changed and episode_index % 2 == 0
            casualty = number_changed and lineage_index % 11 == 0
            records.append(
                {
                    "episode_id": episode_id,
                    "lineage_id": lineage_id,
                    "future_revised": future_revised,
                    "selected_index": selected_index,
                }
            )
            flags[episode_id] = {
                "episode_id": episode_id,
                "number_changed": number_changed,
                "number_dominant_edit": number_dominant,
                "casualty_count_update": casualty,
            }
    return records, flags


def test_grouped_splits_are_deterministic_and_leakage_free() -> None:
    records, flags = _fixture_records()

    first_manifest, first_folds = build_grouped_split_manifest(
        records,
        numeric_flags=flags,
        folds=10,
        seed=17,
    )
    second_manifest, second_folds = build_grouped_split_manifest(
        records,
        numeric_flags=flags,
        folds=10,
        seed=17,
    )

    assert first_manifest == second_manifest
    assert first_folds == second_folds
    assert all(first_manifest["gates"].values())

    all_lineages = {str(record["lineage_id"]) for record in records}
    test_counts = {lineage_id: 0 for lineage_id in all_lineages}
    validation_counts = {lineage_id: 0 for lineage_id in all_lineages}

    for fold in first_folds:
        train = set(fold["train_lineages"])
        validation = set(fold["validation_lineages"])
        test = set(fold["test_lineages"])
        assert not train.intersection(validation)
        assert not train.intersection(test)
        assert not validation.intersection(test)
        assert train.union(validation, test) == all_lineages
        for lineage_id in test:
            test_counts[lineage_id] += 1
        for lineage_id in validation:
            validation_counts[lineage_id] += 1

    assert set(test_counts.values()) == {1}
    assert set(validation_counts.values()) == {1}


def test_seed_changes_outer_assignment() -> None:
    records, flags = _fixture_records()

    first, _ = build_grouped_split_manifest(records, numeric_flags=flags, folds=10, seed=17)
    second, _ = build_grouped_split_manifest(records, numeric_flags=flags, folds=10, seed=23)

    assert first["lineage_to_outer_fold"] != second["lineage_to_outer_fold"]


def test_writes_manifest_fold_files_and_summary(tmp_path: Path) -> None:
    records, flags = _fixture_records()
    manifest, folds = build_grouped_split_manifest(
        records,
        numeric_flags=flags,
        folds=10,
        seed=17,
    )

    write_grouped_split_artifacts(tmp_path, manifest, folds)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "split-summary.json").exists()
    assert (tmp_path / "split-summary.md").exists()
    assert len(list(tmp_path.glob("fold-*.json"))) == 10

    loaded = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["grouping_key"] == "lineage_id"
    assert loaded["policy"]["expected_partition_shares"] == {
        "test": 0.1,
        "train": 0.8,
        "validation": 0.1,
    }
    summary = (tmp_path / "split-summary.md").read_text(encoding="utf-8")
    assert "# Grouped Split Manifest" in summary
    assert "all lineages tested exactly once | PASS" in summary
