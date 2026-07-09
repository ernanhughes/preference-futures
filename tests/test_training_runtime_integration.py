from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from preference_futures.corpora.common import CORPUS_NAMES
from preference_futures.training.common import sha256_file
from preference_futures.training.runtime import prepare_training, run_training_jobs
from preference_futures.training.verify import verify_training_runs

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TRAINING_INTEGRATION") != "1",
    reason="set RUN_TRAINING_INTEGRATION=1 to run the real Transformers smoke test",
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_step2_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpora = tmp_path / "corpora"
    episodes_path = tmp_path / "episodes.jsonl"
    temporal_path = corpora / "temporal-pairs.jsonl"
    split_path = tmp_path / "split-manifest.json"
    episodes = [
        {
            "episode_id": f"episode-{index}",
            "lineage_id": f"nyt::episode-{index}",
            "candidate_a": f"The earlier candidate number {index}.",
            "candidate_b": f"The retained candidate number {index}.",
            "selected_index": index % 2,
            "context_before": "A short context appears before the candidates.",
            "context_after": "A short context appears after the candidates.",
            "future_revised": index % 2 == 0,
            "v2_sentence": "This future field must never enter source training.",
        }
        for index in range(4)
    ]
    temporal = [
        {
            "temporal_pair_id": f"temporal-{index}",
            "lineage_id": f"nyt::external-{index}",
            "earlier_text": f"The report expected event {index} on Monday.",
            "later_text": f"The report expected event {index} on Tuesday.",
            "context_before": "External context before.",
            "context_after": "External context after.",
        }
        for index in range(4)
    ]
    _write_jsonl(episodes_path, episodes)
    _write_jsonl(temporal_path, temporal)
    split_path.write_text("{}\n", encoding="utf-8")

    folds = []
    for fold in range(2):
        folds.append(
            {
                "fold": fold,
                "partitions": {
                    "train": {"records_per_corpus": 4},
                    "validation": {"records_per_corpus": 4},
                },
            }
        )
        for partition in ("train", "validation"):
            for corpus in CORPUS_NAMES:
                records = []
                for index in range(4):
                    source_id = (
                        f"temporal-{index}"
                        if corpus == "temporal_direction"
                        else f"episode-{index}"
                    )
                    record: dict[str, object] = {
                        "corpus_schema_version": 1,
                        "corpus": corpus,
                        "fold": fold,
                        "partition": partition,
                        "source_kind": (
                            "independent_temporal_pair"
                            if corpus == "temporal_direction"
                            else "preference_episode"
                        ),
                        "source_id": source_id,
                        "lineage_id": (
                            f"nyt::external-{index}"
                            if corpus == "temporal_direction"
                            else f"nyt::episode-{index}"
                        ),
                    }
                    if corpus == "language_adaptation":
                        record["mask_indices"] = [2, 6]
                    else:
                        record["target"] = index % 2
                    if corpus == "pair_exposure" and index % 2 == 0:
                        record["candidate_b_source_episode_id"] = (
                            f"episode-{(index + 1) % 4}"
                        )
                    records.append(record)
                _write_jsonl(
                    corpora / f"fold-{fold:02d}" / corpus / f"{partition}.jsonl",
                    records,
                )

    manifest = {
        "corpus_manifest_schema_version": 1,
        "seed": 17,
        "outer_folds": 2,
        "corpora": list(CORPUS_NAMES),
        "sources": {
            "episodes": {
                "path": str(episodes_path.resolve()),
                "bytes": episodes_path.stat().st_size,
                "sha256": sha256_file(episodes_path),
            },
            "split_manifest": {
                "path": str(split_path.resolve()),
                "bytes": split_path.stat().st_size,
                "sha256": sha256_file(split_path),
            },
            "temporal_pairs": {
                "path": str(temporal_path.resolve()),
                "bytes": temporal_path.stat().st_size,
                "sha256": sha256_file(temporal_path),
            },
        },
        "dataset": {
            "episodes": 4,
            "evaluation_lineages": 4,
            "temporal_pairs": 4,
            "temporal_lineages": 4,
        },
        "folds": folds,
        "gates": {"integration_fixture": True},
    }
    corpora.mkdir(parents=True, exist_ok=True)
    (corpora / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (corpora / "temporal-pairs-audit.json").write_text("{}\n", encoding="utf-8")
    return corpora, episodes_path


def test_real_tiny_distilbert_runs_all_six_objectives(tmp_path: Path) -> None:
    corpora, episodes_path = _build_step2_fixture(tmp_path)
    training = tmp_path / "training"
    contract = prepare_training(
        corpora_directory=corpora,
        episodes_path=episodes_path,
        output_directory=training,
        model_id="hf-internal-testing/tiny-random-distilbert",
        model_revision="main",
        seed=17,
        maximum_sequence_length=32,
        batch_size=2,
        update_steps=2,
        learning_rate=5e-5,
        weight_decay=0.0,
        warmup_steps=1,
        gradient_clip_norm=1.0,
        log_every_steps=1,
    )
    assert len(contract["model"]["resolved_revision"]) == 40

    summary = run_training_jobs(
        training,
        folds="0",
        regimes="all",
        device="cpu",
        smoke_steps=1,
    )
    assert len(summary["completed_jobs"]) == 6

    report = verify_training_runs(
        training,
        folds="0",
        regimes="all",
        smoke=True,
    )
    assert report["passed"] is True, report["errors"]
    assert report["observed"]["observed_jobs"] == 6
