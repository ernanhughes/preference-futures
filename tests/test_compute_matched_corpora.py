from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from preference_futures.corpora import (
    build_compute_matched_corpora,
    write_compute_matched_corpora,
)
from preference_futures.corpora.temporal import extract_independent_temporal_pairs
from preference_futures.corpora.verify import verify_compute_matched_corpora


def _episodes() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for lineage_index in range(30):
        lineage = f"nyt::{lineage_index:03d}"
        for episode_index in range(2):
            records.append(
                {
                    "episode_id": f"{lineage}::{episode_index}",
                    "lineage_id": lineage,
                    "candidate_a": f"Earlier candidate {lineage_index} {episode_index}.",
                    "candidate_b": f"Later candidate {lineage_index} {episode_index}.",
                    "selected_index": (lineage_index + episode_index) % 2,
                    "context_before": "Before context.",
                    "context_after": "After context.",
                    "future_revised": episode_index == 0,
                    "v2_version_id": "3",
                }
            )
    return records


def _splits(
    episodes: list[dict[str, object]],
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    lineages = sorted({str(record["lineage_id"]) for record in episodes})
    assignments = {lineage: index % 5 for index, lineage in enumerate(lineages)}
    folds: dict[int, dict[str, object]] = {}
    all_lineages = set(lineages)
    for fold in range(5):
        test = {lineage for lineage, bucket in assignments.items() if bucket == fold}
        validation = {
            lineage for lineage, bucket in assignments.items() if bucket == (fold + 1) % 5
        }
        train = all_lineages - test - validation
        folds[fold] = {
            "fold": fold,
            "train_lineages": sorted(train),
            "validation_lineages": sorted(validation),
            "test_lineages": sorted(test),
        }
    return {"outer_folds": 5, "lineage_to_outer_fold": assignments}, folds


def _temporal_pairs() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for lineage_index in range(60, 160):
        lineage = f"nyt::{lineage_index:03d}"
        for pair_index in range(2):
            records.append(
                {
                    "temporal_pair_id": f"{lineage}::{pair_index}",
                    "lineage_id": lineage,
                    "earlier_text": (
                        f"The earlier temporal sentence {lineage_index} {pair_index}."
                    ),
                    "later_text": f"The later temporal sentence {lineage_index} {pair_index}.",
                    "context_before": "Temporal before.",
                    "context_after": "Temporal after.",
                }
            )
    return records


def test_builds_six_equal_leakage_free_corpora() -> None:
    episodes = _episodes()
    manifest, folds = _splits(episodes)
    corpus_manifest, outputs = build_compute_matched_corpora(
        episodes,
        manifest,
        folds,
        _temporal_pairs(),
        seed=17,
    )

    assert corpus_manifest["corpora"] == [
        "language_adaptation",
        "pair_exposure",
        "temporal_direction",
        "random_label",
        "shuffled_preference",
        "authentic_preference",
    ]
    assert all(corpus_manifest["gates"].values())
    assert set(outputs) == set(range(5))
    for fold, partitions in outputs.items():
        test_lineages = set(folds[fold]["test_lineages"])
        for corpora in partitions.values():
            assert len({len(records) for records in corpora.values()}) == 1
            for name, records in corpora.items():
                assert all("future_revised" not in record for record in records)
                assert all("v2_version_id" not in record for record in records)
                if name != "temporal_direction":
                    assert all(record["lineage_id"] not in test_lineages for record in records)


def test_extracts_temporal_pairs_from_external_lineages(tmp_path: Path) -> None:
    database = tmp_path / "nyt-matched-sentences.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE split_sentences "
        "(entry_id TEXT, version INTEGER, sent_idx INTEGER, sentence TEXT)"
    )
    rows = []
    for article in ("eval", "external-a", "external-b"):
        rows.extend(
            [
                (article, 1, 0, "The committee said it would meet on Monday afternoon."),
                (article, 1, 1, f"The stable sentence for {article} remains unchanged."),
                (article, 2, 0, "The committee said it would meet on Tuesday afternoon."),
                (article, 2, 1, f"The stable sentence for {article} remains unchanged."),
                (article, 3, 0, "The committee said it would meet on Wednesday afternoon."),
                (article, 3, 1, f"The stable sentence for {article} remains unchanged."),
            ]
        )
    connection.executemany("INSERT INTO split_sentences VALUES (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()

    pairs, audit = extract_independent_temporal_pairs(
        database,
        excluded_lineages={"nyt::eval"},
        source_name="nyt",
        target_pairs=2,
        seed=17,
        max_articles=10,
    )

    assert len(pairs) == 2
    assert all(record["lineage_id"] != "nyt::eval" for record in pairs)
    assert audit["gates"]["temporal_lineages_disjoint_from_evaluation"] is True
    assert audit["gates"]["target_pair_count_reached"] is True
    json.dumps(audit)


def test_persisted_verifier_accepts_written_corpora(tmp_path: Path) -> None:
    episodes = _episodes()
    split_manifest, folds = _splits(episodes)
    episodes_path = tmp_path / "episodes.jsonl"
    split_path = tmp_path / "split-manifest.json"
    temporal_path = tmp_path / "temporal-source.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(record) + "\n" for record in episodes), encoding="utf-8"
    )
    split_path.write_text(json.dumps(split_manifest), encoding="utf-8")
    temporal = _temporal_pairs()
    temporal_path.write_text(
        "".join(json.dumps(record) + "\n" for record in temporal), encoding="utf-8"
    )
    manifest, outputs = build_compute_matched_corpora(
        episodes,
        split_manifest,
        folds,
        temporal,
        seed=17,
        episodes_path=episodes_path,
        split_manifest_path=split_path,
        temporal_pairs_path=temporal_path,
    )
    output = tmp_path / "corpora"
    write_compute_matched_corpora(output, manifest, outputs)
    (output / "temporal-pairs.jsonl").write_text(
        temporal_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "temporal-pairs-audit.json").write_text("{}\n", encoding="utf-8")

    report = verify_compute_matched_corpora(output)

    assert report["passed"] is True
    assert all(report["checks"].values())
