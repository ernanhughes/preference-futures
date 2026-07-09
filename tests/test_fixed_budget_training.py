from __future__ import annotations

import json
from pathlib import Path

from preference_futures.corpora.common import CORPUS_NAMES
from preference_futures.training.common import sha256_file
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
    serialise_episode,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_source_store_redacts_future_fields_and_materializes_controls(tmp_path: Path) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    temporal_path = tmp_path / "temporal.jsonl"
    _write_jsonl(
        episodes_path,
        [
            {
                "episode_id": "one",
                "lineage_id": "nyt::one",
                "candidate_a": "The earlier sentence.",
                "candidate_b": "The retained sentence.",
                "context_before": "Before.",
                "context_after": "After.",
                "future_revised": True,
                "v2_sentence": "Forbidden future text.",
            },
            {
                "episode_id": "two",
                "lineage_id": "nyt::two",
                "candidate_a": "Another earlier sentence.",
                "candidate_b": "A donor retained sentence.",
                "context_before": "Other before.",
                "context_after": "Other after.",
                "future_revised": False,
            },
        ],
    )
    _write_jsonl(
        temporal_path,
        [
            {
                "temporal_pair_id": "temporal-one",
                "lineage_id": "nyt::external",
                "earlier_text": "It was expected Monday.",
                "later_text": "It was expected Tuesday.",
                "context_before": "External before.",
                "context_after": "External after.",
            }
        ],
    )

    store = load_source_store(episodes_path, temporal_path)
    assert "future_revised" not in store.episodes["one"]
    assert "v2_sentence" not in store.episodes["one"]

    authentic = materialize_record(
        {"corpus": "authentic_preference", "source_id": "one", "target": 1},
        store,
    )
    assert isinstance(authentic, ClassificationExample)
    assert authentic.target == 1
    assert "The retained sentence." in authentic.text
    assert authentic.text.startswith("[CONTEXT_BEFORE]\n")

    negative_pair = materialize_record(
        {
            "corpus": "pair_exposure",
            "source_id": "one",
            "target": 0,
            "candidate_b_source_episode_id": "two",
        },
        store,
    )
    assert isinstance(negative_pair, ClassificationExample)
    assert "A donor retained sentence." in negative_pair.text

    temporal = materialize_record(
        {"corpus": "temporal_direction", "source_id": "temporal-one", "target": 0},
        store,
    )
    assert isinstance(temporal, ClassificationExample)
    assert temporal.text.index("It was expected Tuesday.") < temporal.text.index(
        "It was expected Monday."
    )

    words = serialise_episode(store.episodes["one"]).split()
    language = materialize_record(
        {
            "corpus": "language_adaptation",
            "source_id": "one",
            "mask_indices": [1, len(words) - 1],
        },
        store,
    )
    assert isinstance(language, MaskedLanguageExample)
    assert language.mask_word_indices == (1, len(words) - 1)
    assert language.words == tuple(words)


def test_training_batches_are_fixed_size_deterministic_and_cycle() -> None:
    first = deterministic_training_batches(5, batch_size=4, update_steps=4, seed=17)
    second = deterministic_training_batches(5, batch_size=4, update_steps=4, seed=17)
    assert first == second
    assert len(first) == 4
    assert all(len(batch) == 4 for batch in first)
    assert all(0 <= index < 5 for batch in first for index in batch)
    assert len([index for batch in first for index in batch]) == 16


def test_builds_and_validates_frozen_training_contract(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    episodes_path = tmp_path / "episodes.jsonl"
    temporal_path = corpora / "temporal-pairs.jsonl"
    split_path = tmp_path / "split-manifest.json"
    _write_jsonl(
        episodes_path,
        [
            {
                "episode_id": "episode-one",
                "lineage_id": "nyt::one",
                "candidate_a": "Earlier.",
                "candidate_b": "Later.",
                "selected_index": 1,
                "context_before": "Before.",
                "context_after": "After.",
                "future_revised": False,
            }
        ],
    )
    _write_jsonl(
        temporal_path,
        [
            {
                "temporal_pair_id": "temporal-one",
                "lineage_id": "nyt::external",
                "earlier_text": "Earlier external.",
                "later_text": "Later external.",
            }
        ],
    )
    split_path.write_text("{}\n", encoding="utf-8")

    fold_summaries = []
    for fold in range(2):
        fold_summaries.append(
            {
                "fold": fold,
                "partitions": {
                    "train": {"records_per_corpus": 1},
                    "validation": {"records_per_corpus": 1},
                },
            }
        )
        for partition in ("train", "validation"):
            for corpus in CORPUS_NAMES:
                record: dict[str, object] = {
                    "corpus": corpus,
                    "fold": fold,
                    "partition": partition,
                    "source_kind": (
                        "independent_temporal_pair"
                        if corpus == "temporal_direction"
                        else "preference_episode"
                    ),
                    "source_id": (
                        "temporal-one" if corpus == "temporal_direction" else "episode-one"
                    ),
                    "lineage_id": (
                        "nyt::external" if corpus == "temporal_direction" else "nyt::one"
                    ),
                }
                if corpus == "language_adaptation":
                    record["mask_indices"] = [0]
                else:
                    record["target"] = 1
                _write_jsonl(
                    corpora / f"fold-{fold:02d}" / corpus / f"{partition}.jsonl",
                    [record],
                )

    manifest = {
        "outer_folds": 2,
        "corpora": list(CORPUS_NAMES),
        "sources": {
            "episodes": {
                "path": str(episodes_path.resolve()),
                "sha256": sha256_file(episodes_path),
            },
            "split_manifest": {
                "path": str(split_path.resolve()),
                "sha256": sha256_file(split_path),
            },
            "temporal_pairs": {
                "path": str(temporal_path.resolve()),
                "sha256": sha256_file(temporal_path),
            },
        },
        "gates": {"fixture": True},
        "folds": fold_summaries,
    }
    corpora.mkdir(parents=True, exist_ok=True)
    (corpora / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (corpora / "temporal-pairs-audit.json").write_text("{}\n", encoding="utf-8")

    snapshot = tmp_path / "base-snapshot"
    (snapshot / "encoder").mkdir(parents=True)
    (snapshot / "tokenizer").mkdir(parents=True)
    (snapshot / "encoder" / "model.safetensors").write_bytes(b"encoder")
    (snapshot / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "training"

    contract = build_training_contract(
        corpora_directory=corpora,
        episodes_path=episodes_path,
        output_directory=output,
        base_snapshot_directory=snapshot,
        model_metadata={
            "model_id": "fixture-model",
            "resolved_revision": "fixture-revision",
        },
        update_steps=3,
        batch_size=2,
        maximum_sequence_length=8,
        warmup_steps=1,
        log_every_steps=1,
    )
    assert contract["expected_training_jobs"] == 12
    assert contract["optimisation"]["padded_token_positions_per_job"] == 48
    assert all(contract["gates"].values())

    write_training_contract(output, contract)
    validate_training_contract(contract)
    assert (output / "contract.json").exists()
    assert (output / "training-plan.md").exists()
