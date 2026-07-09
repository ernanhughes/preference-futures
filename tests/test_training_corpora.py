from __future__ import annotations

import json
from pathlib import Path

import pytest

from preference_futures.corpora import build_training_corpora, write_training_corpora
from preference_futures.splits import build_grouped_split_manifest


def _record(index: int) -> dict[str, object]:
    selected_index = index % 2
    return {
        "schema_version": 1,
        "episode_id": f"episode-{index:03d}",
        "lineage_id": f"nyt::{index // 2:03d}",
        "candidate_a": f"Earlier sentence {index}.",
        "candidate_b": f"Later sentence {index}.",
        "selected_index": selected_index,
        "future_revised": index % 3 == 0,
        "v0_version_id": "1",
        "v1_version_id": "2",
        "v2_version_id": "3",
        "selected_sentence_index": index,
        "context_before": "Before context.",
        "context_after": "After context.",
        "sentence_position": 0.5,
        "edit_similarity": 0.8,
        "lexical_jaccard": 0.7,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[list[dict[str, object]], Path, Path]:
    records = [_record(index) for index in range(60)]
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(episodes_path, records)
    split_manifest, _ = build_grouped_split_manifest(
        records,
        folds=5,
        seed=17,
        episodes_path=episodes_path,
    )
    manifest_path = tmp_path / "split-manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, sort_keys=True), encoding="utf-8")
    return records, episodes_path, manifest_path


def test_builds_compute_matched_corpora_with_same_partition_counts(tmp_path: Path) -> None:
    records, episodes_path, manifest_path = _fixture(tmp_path)
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest, corpora = build_training_corpora(
        records,
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=manifest_path,
        seed=17,
    )

    assert all(manifest["gates"].values())
    assert manifest["source_checks"]["episodes_sha256_matches_split_manifest"] is True
    assert set(manifest["corpus_names"]) == {
        "authentic_preference",
        "language_modeling_control",
        "pair_exposure_control",
        "random_label_control",
        "shuffled_preference_control",
        "temporal_direction_control",
    }

    for fold in range(5):
        for partition in ("train", "validation", "test"):
            counts = {
                len(corpora[name][fold][partition]) for name in manifest["corpus_names"]
            }
            token_counts = {
                manifest["corpora"][name]["folds"][f"fold-{fold:02d}"][partition][
                    "input_tokens_whitespace"
                ]
                for name in manifest["corpus_names"]
            }
            assert len(counts) == 1
            assert len(token_counts) == 1


def test_corpus_records_redact_future_labels(tmp_path: Path) -> None:
    records, episodes_path, manifest_path = _fixture(tmp_path)
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _, corpora = build_training_corpora(
        records,
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=manifest_path,
        seed=17,
    )

    sample = corpora["authentic_preference"][0]["train"][0]
    assert "future_revised" not in sample
    assert sample["label_name"] == "selected_candidate_index"
    assert sample["label"] in (0, 1)

    for record in corpora["authentic_preference"][0]["test"]:
        assert record["source_training_allowed"] is False

    unlabelled = corpora["pair_exposure_control"][0]["train"][0]
    assert unlabelled["label_name"] is None
    assert unlabelled["label"] is None


def test_shuffled_labels_are_deterministic(tmp_path: Path) -> None:
    records, episodes_path, manifest_path = _fixture(tmp_path)
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    first_manifest, first = build_training_corpora(
        records,
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=manifest_path,
        seed=17,
    )
    second_manifest, second = build_training_corpora(
        records,
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=manifest_path,
        seed=17,
    )

    assert first_manifest == second_manifest
    assert first == second

    for fold in range(5):
        for partition in ("train", "validation", "test"):
            authentic_labels = sorted(
                record["label"] for record in first["authentic_preference"][fold][partition]
            )
            shuffled_labels = sorted(
                record["label"] for record in first["shuffled_preference_control"][fold][partition]
            )
            assert shuffled_labels == authentic_labels


def test_rejects_changed_episode_file(tmp_path: Path) -> None:
    records, episodes_path, manifest_path = _fixture(tmp_path)
    original = episodes_path.read_text(encoding="utf-8")
    episodes_path.write_text(original + "\n", encoding="utf-8")
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="episodes SHA-256"):
        build_training_corpora(
            records,
            split_manifest,
            episodes_path=episodes_path,
            split_manifest_path=manifest_path,
            seed=17,
        )


def test_writes_corpus_manifest_summary_and_fold_files(tmp_path: Path) -> None:
    records, episodes_path, manifest_path = _fixture(tmp_path)
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest, corpora = build_training_corpora(
        records,
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=manifest_path,
        seed=17,
    )

    output = tmp_path / "corpora"
    write_training_corpora(output, manifest, corpora)

    assert (output / "corpus-manifest.json").exists()
    assert (output / "corpus-summary.md").exists()
    assert (output / "authentic_preference" / "fold-00" / "train.jsonl").exists()
    assert (output / "random_label_control" / "fold-04" / "test.jsonl").exists()
    summary = (output / "corpus-summary.md").read_text(encoding="utf-8")
    assert "# Compute-Matched Training Corpora" in summary
    assert "authentic_preference" in summary
