"""Build auditable, fold-locked source-task corpora for representation training."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.corpora.common import (
    CORPUS_NAMES,
    FORBIDDEN_FUTURE_KEYS,
    assert_partition,
    load_json,
    load_jsonl,
    positive_int,
    source_metadata,
    string_set,
    validate_episodes,
    validate_temporal_pairs,
)
from preference_futures.corpora.controls import (
    assign_temporal_lineages,
    build_partition_corpora,
    record_exposure_tokens,
)


def build_compute_matched_corpora(
    episodes: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    fold_documents: Mapping[int, Mapping[str, Any]],
    temporal_pairs: Sequence[Mapping[str, Any]],
    *,
    seed: int = 17,
    episodes_path: Path | None = None,
    split_manifest_path: Path | None = None,
    temporal_pairs_path: Path | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, list[dict[str, Any]]]]]]:
    """Build six source-task corpora with frozen fold membership and record budgets."""

    episode_by_id = validate_episodes(episodes)
    evaluation_lineages = {str(record["lineage_id"]) for record in episodes}
    temporal_by_id = validate_temporal_pairs(temporal_pairs, evaluation_lineages)
    outer_folds = positive_int(split_manifest.get("outer_folds"), "outer_folds")
    assignments = split_manifest.get("lineage_to_outer_fold")
    if not isinstance(assignments, Mapping):
        raise ValueError("split manifest requires lineage_to_outer_fold")
    if set(map(str, assignments)) != evaluation_lineages:
        raise ValueError("split manifest lineage assignments do not match episode lineages")
    if set(fold_documents) != set(range(outer_folds)):
        raise ValueError("fold documents must cover every outer fold exactly once")

    temporal_assignments = assign_temporal_lineages(temporal_pairs, outer_folds, seed)
    outputs: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    fold_summaries: list[dict[str, Any]] = []

    for fold in range(outer_folds):
        document = fold_documents[fold]
        train_lineages = string_set(document.get("train_lineages"), f"fold {fold} train")
        validation_lineages = string_set(
            document.get("validation_lineages"), f"fold {fold} validation"
        )
        test_lineages = string_set(document.get("test_lineages"), f"fold {fold} test")
        assert_partition(train_lineages, validation_lineages, test_lineages, evaluation_lineages)

        train_episodes = [
            record for record in episodes if str(record["lineage_id"]) in train_lineages
        ]
        validation_episodes = [
            record for record in episodes if str(record["lineage_id"]) in validation_lineages
        ]

        temporal_validation_bucket = (fold + 1) % outer_folds
        temporal_test_bucket = fold
        temporal_train_pool = [
            record
            for record in temporal_pairs
            if temporal_assignments[str(record["lineage_id"])]
            not in {temporal_test_bucket, temporal_validation_bucket}
        ]
        temporal_validation_pool = [
            record
            for record in temporal_pairs
            if temporal_assignments[str(record["lineage_id"])] == temporal_validation_bucket
        ]

        partitions: dict[str, dict[str, list[dict[str, Any]]]] = {}
        partition_summaries: dict[str, Any] = {}
        for partition_name, partition_episodes, temporal_pool in (
            ("train", train_episodes, temporal_train_pool),
            ("validation", validation_episodes, temporal_validation_pool),
        ):
            corpora = build_partition_corpora(
                partition_episodes,
                temporal_pool,
                seed=seed,
                fold=fold,
                partition=partition_name,
            )
            partitions[partition_name] = corpora
            partition_summaries[partition_name] = _summarise_partition(
                corpora,
                episode_by_id,
                temporal_by_id,
            )
        outputs[fold] = partitions
        fold_summaries.append(
            {
                "fold": fold,
                "train_lineages": len(train_lineages),
                "validation_lineages": len(validation_lineages),
                "test_lineages": len(test_lineages),
                "partitions": partition_summaries,
            }
        )

    gates = _build_gates(outputs, episode_by_id, temporal_by_id, fold_documents)
    manifest = {
        "corpus_manifest_schema_version": 1,
        "seed": seed,
        "outer_folds": outer_folds,
        "corpora": list(CORPUS_NAMES),
        "sources": {
            "episodes": source_metadata(episodes_path),
            "split_manifest": source_metadata(split_manifest_path),
            "temporal_pairs": source_metadata(temporal_pairs_path),
        },
        "dataset": {
            "episodes": len(episodes),
            "evaluation_lineages": len(evaluation_lineages),
            "temporal_pairs": len(temporal_pairs),
            "temporal_lineages": len({str(row["lineage_id"]) for row in temporal_pairs}),
        },
        "temporal_policy": {
            "evaluation_lineage_overlap_allowed": False,
            "assignment": "deterministic lineage-grouped balanced outer buckets",
            "test_bucket": "fold i (unused for source training)",
            "validation_bucket": "fold (i + 1) mod K",
            "training_buckets": "all remaining buckets",
        },
        "compute_matching_policy": {
            "source_records_per_partition": "exactly equal across all six trained regimes",
            "optimizer_steps": "must be identical and fixed in Step 3",
            "batch_size": "must be identical in Step 3",
            "maximum_sequence_length": "must be identical in Step 3",
            "padding": "pad every source-task batch to the same frozen maximum length",
            "tokenizer": "must be identical and frozen in Step 3",
            "checkpoint_selection": "fixed update checkpoints; no task-specific early stopping",
            "note": (
                "Step 2 matches source-record budgets and records exposure estimates. "
                "Exact compute matching is enforced by the Step 3 trainer, not inferred from file size."
            ),
        },
        "identification_note": {
            "exact_pair_temporal_target_equals_authentic_target": True,
            "reason": (
                "In a V0-to-V1 revision pair, the editor-retained sentence is also the later "
                "sentence. Training temporal direction on those exact pairs would duplicate the "
                "authentic labels rather than provide an independent control."
            ),
            "resolution": (
                "The temporal-direction corpus is therefore drawn from separate NewsEdits article "
                "lineages that never appear in future evaluation, while pair-exposure and language "
                "adaptation controls retain the exact evaluation-domain texts."
            ),
        },
        "folds": fold_summaries,
        "gates": gates,
        "warnings": [
            "Future labels and V2 fields are forbidden from every source-task JSONL record.",
            (
                "The temporal-direction pool is drawn from article lineages disjoint from "
                "all future-evaluation lineages."
            ),
            "Do not alter corpus assignments after observing downstream future-probe results.",
        ],
    }
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise ValueError(f"compute-matched corpus gates failed: {failed}")
    return manifest, outputs


def write_compute_matched_corpora(
    output_directory: Path,
    manifest: Mapping[str, Any],
    outputs: Mapping[int, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> None:
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "corpus-summary.md").write_text(
        render_corpus_summary_markdown(manifest), encoding="utf-8"
    )
    for fold, partitions in outputs.items():
        fold_dir = output / f"fold-{fold:02d}"
        for partition, corpora in partitions.items():
            for corpus_name, records in corpora.items():
                path = fold_dir / corpus_name / f"{partition}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")


def render_corpus_summary_markdown(manifest: Mapping[str, Any]) -> str:
    dataset = manifest["dataset"]
    lines = [
        "# Compute-Matched Source Corpora",
        "",
        "## Dataset",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Preference episodes | {dataset['episodes']:,} |",
        f"| Evaluation lineages | {dataset['evaluation_lineages']:,} |",
        f"| Independent temporal pairs | {dataset['temporal_pairs']:,} |",
        f"| Independent temporal lineages | {dataset['temporal_lineages']:,} |",
        "",
        "## Corpus regimes",
        "",
    ]
    lines.extend(f"- `{name}`" for name in manifest["corpora"])
    lines.extend(
        [
            "",
            "## Fold budgets",
            "",
            (
                "| Fold | Partition | Records per corpus | Min whitespace tokens | "
                "Max whitespace tokens |"
            ),
            "|---:|---|---:|---:|---:|",
        ]
    )
    for fold in manifest["folds"]:
        for partition, values in fold["partitions"].items():
            lines.append(
                "| {fold} | {partition} | {records:,} | {minimum:,} | {maximum:,} |".format(
                    fold=fold["fold"],
                    partition=partition,
                    records=values["records_per_corpus"],
                    minimum=values["minimum_exposure_tokens"],
                    maximum=values["maximum_exposure_tokens"],
                )
            )
    lines.extend(["", "## Gates", "", "| Gate | Result |", "|---|---|"])
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in manifest["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Experimental consequence",
            "",
            "Step 3 must use the same encoder checkpoint, tokenizer, batch size, maximum sequence",
            "length, optimiser, learning-rate schedule, update count and fixed checkpoint rule for",
            "all six trained regimes. The untouched generic encoder remains the seventh arm.",
            "",
        ]
    )
    return "\n".join(lines)


def _summarise_partition(
    corpora: Mapping[str, Sequence[Mapping[str, Any]]],
    episodes: Mapping[str, Mapping[str, Any]],
    temporal_pairs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {name: len(records) for name, records in corpora.items()}
    exposures = {
        name: sum(record_exposure_tokens(record, episodes, temporal_pairs) for record in records)
        for name, records in corpora.items()
    }
    return {
        "records_per_corpus": next(iter(counts.values())),
        "record_counts": counts,
        "exposure_tokens": exposures,
        "minimum_exposure_tokens": min(exposures.values()),
        "maximum_exposure_tokens": max(exposures.values()),
        "shared_step_budget": next(iter(counts.values())),
    }


def _build_gates(
    outputs: Mapping[int, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    episodes: Mapping[str, Mapping[str, Any]],
    temporal_pairs: Mapping[str, Mapping[str, Any]],
    fold_documents: Mapping[int, Mapping[str, Any]],
) -> dict[str, bool]:
    equal_counts = True
    no_test_leakage = True
    no_future_fields = True
    random_balanced = True
    pair_balanced = True
    pair_donors_cross_lineage = True
    shuffled_preserves_labels = True
    shuffled_donors_cross_lineage = True
    temporal_external = True
    temporal_balanced = True
    evaluation_lineages = {str(value["lineage_id"]) for value in episodes.values()}

    for fold, partitions in outputs.items():
        test_lineages = set(map(str, fold_documents[fold]["test_lineages"]))
        for corpora in partitions.values():
            counts = {len(records) for records in corpora.values()}
            equal_counts = equal_counts and len(counts) == 1
            for corpus_name, records in corpora.items():
                no_future_fields = no_future_fields and all(
                    not FORBIDDEN_FUTURE_KEYS.intersection(record) for record in records
                )
                if corpus_name != "temporal_direction":
                    no_test_leakage = no_test_leakage and all(
                        str(record["lineage_id"]) not in test_lineages for record in records
                    )
            authentic = corpora["authentic_preference"]
            random_records = corpora["random_label"]
            shuffled = corpora["shuffled_preference"]
            pair_records = corpora["pair_exposure"]
            temporal = corpora["temporal_direction"]

            random_counts = Counter(int(row["target"]) for row in random_records)
            random_balanced = random_balanced and abs(random_counts[0] - random_counts[1]) <= 1
            pair_counts = Counter(int(row["target"]) for row in pair_records)
            pair_balanced = pair_balanced and abs(pair_counts[0] - pair_counts[1]) <= 1
            temporal_counts = Counter(int(row["target"]) for row in temporal)
            temporal_balanced = temporal_balanced and (
                abs(temporal_counts[0] - temporal_counts[1]) <= 1
            )

            for row in pair_records:
                if int(row["target"]) == 0:
                    source = episodes[str(row["source_id"])]
                    donor = episodes[str(row["candidate_b_source_episode_id"])]
                    pair_donors_cross_lineage = pair_donors_cross_lineage and (
                        str(source["lineage_id"]) != str(donor["lineage_id"])
                    )
            authentic_labels = Counter(int(row["target"]) for row in authentic)
            shuffled_labels = Counter(int(row["target"]) for row in shuffled)
            shuffled_preserves_labels = shuffled_preserves_labels and (
                authentic_labels == shuffled_labels
            )
            for row in shuffled:
                source = episodes[str(row["source_id"])]
                donor = episodes[str(row["label_source_episode_id"])]
                shuffled_donors_cross_lineage = shuffled_donors_cross_lineage and (
                    str(source["lineage_id"]) != str(donor["lineage_id"])
                )
            temporal_external = temporal_external and all(
                str(row["lineage_id"]) not in evaluation_lineages
                and str(row["source_id"]) in temporal_pairs
                for row in temporal
            )

    return {
        "all_six_corpora_have_equal_record_counts": equal_counts,
        "no_preference_source_record_uses_test_lineages": no_test_leakage,
        "no_source_task_record_contains_future_fields": no_future_fields,
        "random_labels_are_balanced": random_balanced,
        "pair_exposure_labels_are_balanced": pair_balanced,
        "pair_exposure_negative_donors_cross_lineages": pair_donors_cross_lineage,
        "shuffled_preferences_preserve_label_counts": shuffled_preserves_labels,
        "shuffled_preference_donors_cross_lineages": shuffled_donors_cross_lineage,
        "temporal_pairs_are_external_to_evaluation_lineages": temporal_external,
        "temporal_direction_labels_are_balanced": temporal_balanced,
    }


__all__ = [
    "CORPUS_NAMES",
    "FORBIDDEN_FUTURE_KEYS",
    "build_compute_matched_corpora",
    "load_json",
    "load_jsonl",
    "render_corpus_summary_markdown",
    "write_compute_matched_corpora",
]
