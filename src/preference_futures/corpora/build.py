"""Build deterministic compute-matched source-task corpora for transfer experiments."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_CORPUS_SCHEMA_VERSION = 1

PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    name: str
    family: str
    objective: str
    label_name: str | None
    description: str


CORPUS_SPECS = (
    CorpusSpec(
        name="authentic_preference",
        family="preference",
        objective="predict_editor_selected_candidate",
        label_name="selected_candidate_index",
        description="Authentic V0/V1 preference supervision: predict which candidate was retained.",
    ),
    CorpusSpec(
        name="language_modeling_control",
        family="control",
        objective="same_text_exposure_without_pair_label",
        label_name=None,
        description="Unlabelled exposure to the same serialized pair/context text.",
    ),
    CorpusSpec(
        name="pair_exposure_control",
        family="control",
        objective="same_pair_exposure_without_selection_label",
        label_name=None,
        description="Revision-pair exposure with no candidate-selection target.",
    ),
    CorpusSpec(
        name="temporal_direction_control",
        family="control",
        objective="predict_newer_candidate",
        label_name="newer_candidate_index",
        description="Predict which candidate is the later revision; in this dataset that is V1.",
    ),
    CorpusSpec(
        name="random_label_control",
        family="control",
        objective="predict_deterministic_random_candidate",
        label_name="random_candidate_index",
        description="Preference-shaped optimization with deterministic random labels.",
    ),
    CorpusSpec(
        name="shuffled_preference_control",
        family="control",
        objective="predict_partition_shuffled_preference_label",
        label_name="shuffled_selected_candidate_index",
        description="Preference-shaped labels shuffled within each fold partition.",
    ),
)


@dataclass(frozen=True, slots=True)
class AssignedEpisode:
    record: Mapping[str, Any]
    fold: int
    partition: str


def build_training_corpora(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    episodes_path: Path | None = None,
    split_manifest_path: Path | None = None,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, list[dict[str, Any]]]]]]:
    """Build in-memory corpus records and a manifest.

    Step 2 intentionally does not train a model. It freezes the source-task
    examples consumed by later representation training so that authentic
    preference learning and its controls differ by supervision, not by split,
    row population or input text.
    """

    if not records:
        raise ValueError("records must not be empty")

    source_checks = _validate_sources(
        split_manifest,
        episodes_path=episodes_path,
        split_manifest_path=split_manifest_path,
    )
    assignments = _lineage_assignments(split_manifest)
    folds = _outer_folds(split_manifest)
    assigned = _assign_records(records, assignments=assignments, folds=folds)
    label_maps = _label_maps(assigned, seed=seed)

    corpora: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    corpus_stats: dict[str, Any] = {}
    for spec in CORPUS_SPECS:
        by_fold: dict[int, dict[str, list[dict[str, Any]]]] = {}
        stats_by_fold: dict[str, Any] = {}
        for fold in range(folds):
            by_partition: dict[str, list[dict[str, Any]]] = {}
            stats_by_partition: dict[str, Any] = {}
            for partition in PARTITIONS:
                partition_records = [
                    item for item in assigned if item.fold == fold and item.partition == partition
                ]
                output_records = [
                    _corpus_record(
                        item.record,
                        spec=spec,
                        fold=fold,
                        partition=partition,
                        label=label_maps.get(spec.name, {}).get((fold, partition), {}).get(
                            str(item.record["episode_id"])
                        ),
                    )
                    for item in partition_records
                ]
                by_partition[partition] = output_records
                stats_by_partition[partition] = _partition_stats(
                    output_records,
                    source_records=[item.record for item in partition_records],
                )
            by_fold[fold] = by_partition
            stats_by_fold[f"fold-{fold:02d}"] = stats_by_partition
        corpora[spec.name] = by_fold
        corpus_stats[spec.name] = {
            "family": spec.family,
            "objective": spec.objective,
            "label_name": spec.label_name,
            "description": spec.description,
            "folds": stats_by_fold,
        }

    manifest = {
        "training_corpus_schema_version": TRAINING_CORPUS_SCHEMA_VERSION,
        "seed": seed,
        "outer_folds": folds,
        "partitions": list(PARTITIONS),
        "corpus_names": [spec.name for spec in CORPUS_SPECS],
        "source_checks": source_checks,
        "sources": {
            "episodes": _source_metadata(episodes_path),
            "split_manifest": _source_metadata(split_manifest_path),
        },
        "split_identity": _split_identity(split_manifest),
        "totals": _global_totals(records),
        "corpora": corpus_stats,
    }
    manifest["gates"] = _build_gates(manifest)
    manifest["warnings"] = _build_warnings(manifest)
    return manifest, corpora


def write_training_corpora(
    output_directory: Path,
    manifest: Mapping[str, Any],
    corpora: Mapping[str, Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> None:
    """Write corpus JSONL files and summary artifacts."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for corpus_name, by_fold in corpora.items():
        for fold, by_partition in by_fold.items():
            fold_dir = output / corpus_name / f"fold-{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            for partition, records in by_partition.items():
                _write_jsonl(fold_dir / f"{partition}.jsonl", records)

    (output / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "corpus-summary.md").write_text(
        render_training_corpus_markdown(manifest),
        encoding="utf-8",
    )


def render_training_corpus_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact review summary for Step 2."""

    totals = manifest["totals"]
    lines = [
        "# Compute-Matched Training Corpora",
        "",
        "## Dataset",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Episodes | {totals['episodes']:,} |",
        f"| Article lineages | {totals['lineages']:,} |",
        f"| Future revised | {totals['future_revised']:,} |",
        f"| Future-revision rate | {totals['future_revised_rate']:.4f} |",
        f"| Outer folds | {manifest['outer_folds']} |",
        f"| Corpora | {len(manifest['corpus_names'])} |",
        "",
        "## Corpora",
        "",
        "| Corpus | Family | Label | Objective |",
        "|---|---|---|---|",
    ]
    for name in manifest["corpus_names"]:
        corpus = manifest["corpora"][name]
        label = corpus["label_name"] if corpus["label_name"] is not None else "none"
        lines.append(f"| {name} | {corpus['family']} | {label} | {corpus['objective']} |")

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in manifest["gates"].items()
    )

    lines.extend(["", "## Fold 0 example counts", ""])
    lines.extend(["| Corpus | Train | Validation | Test |", "|---|---:|---:|---:|"])
    for name in manifest["corpus_names"]:
        fold = manifest["corpora"][name]["folds"]["fold-00"]
        lines.append(
            "| {name} | {train:,} | {validation:,} | {test:,} |".format(
                name=name,
                train=fold["train"]["records"],
                validation=fold["validation"]["records"],
                test=fold["test"]["records"],
            )
        )

    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Experimental consequence",
            "",
            "Later representation training must consume these exact source-task corpora.",
            "Future labels are not written into corpus records; they remain reserved for the",
            "downstream frozen-representation probe stage.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_sources(
    split_manifest: Mapping[str, Any],
    *,
    episodes_path: Path | None,
    split_manifest_path: Path | None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if episodes_path is not None:
        expected = (
            split_manifest.get("sources", {})
            .get("episodes", {})
            .get("sha256")
        )
        observed = _sha256(episodes_path)
        checks["episodes_sha256_matches_split_manifest"] = observed == expected
        if expected is not None and observed != expected:
            raise ValueError("episodes SHA-256 does not match the frozen split manifest")
    else:
        checks["episodes_sha256_matches_split_manifest"] = None

    checks["split_manifest_sha256"] = _sha256(split_manifest_path) if split_manifest_path else None
    return checks


def _lineage_assignments(split_manifest: Mapping[str, Any]) -> dict[str, int]:
    value = split_manifest.get("lineage_to_outer_fold")
    if not isinstance(value, Mapping) or not value:
        raise ValueError("split manifest requires lineage_to_outer_fold")
    assignments: dict[str, int] = {}
    for lineage_id, fold in value.items():
        if type(fold) is not int:
            raise TypeError(f"fold assignment for {lineage_id!r} must be an int")
        assignments[str(lineage_id)] = fold
    return assignments


def _outer_folds(split_manifest: Mapping[str, Any]) -> int:
    folds = split_manifest.get("outer_folds")
    if type(folds) is not int or folds < 3:
        raise ValueError("split manifest outer_folds must be an integer of at least 3")
    return folds


def _assign_records(
    records: Sequence[Mapping[str, Any]],
    *,
    assignments: Mapping[str, int],
    folds: int,
) -> list[AssignedEpisode]:
    assigned: list[AssignedEpisode] = []
    for fold in range(folds):
        validation_bucket = (fold + 1) % folds
        for record in records:
            lineage_id = str(record["lineage_id"])
            if lineage_id not in assignments:
                raise ValueError(f"episode lineage {lineage_id!r} is absent from split manifest")
            bucket = assignments[lineage_id]
            if bucket == fold:
                partition = "test"
            elif bucket == validation_bucket:
                partition = "validation"
            else:
                partition = "train"
            assigned.append(AssignedEpisode(record=record, fold=fold, partition=partition))
    return assigned


def _label_maps(
    assigned: Sequence[AssignedEpisode],
    *,
    seed: int,
) -> dict[str, dict[tuple[int, str], dict[str, int]]]:
    random_labels: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    shuffled_labels: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    grouped: dict[tuple[int, str], list[AssignedEpisode]] = defaultdict(list)
    for item in assigned:
        key = (item.fold, item.partition)
        grouped[key].append(item)
        random_labels[key][str(item.record["episode_id"])] = _stable_bit(
            seed,
            "random_label_control",
            str(item.fold),
            item.partition,
            str(item.record["episode_id"]),
        )

    for key, items in grouped.items():
        source = sorted(items, key=lambda item: str(item.record["episode_id"]))
        labels = [int(item.record["selected_index"]) for item in source]
        target = sorted(
            items,
            key=lambda item: _stable_hex(
                seed,
                "shuffled_preference_control",
                str(key[0]),
                key[1],
                str(item.record["episode_id"]),
            ),
        )
        if len(labels) > 1:
            labels = labels[1:] + labels[:1]
        for item, label in zip(target, labels, strict=True):
            shuffled_labels[key][str(item.record["episode_id"])] = label

    return {
        "random_label_control": dict(random_labels),
        "shuffled_preference_control": dict(shuffled_labels),
    }


def _corpus_record(
    record: Mapping[str, Any],
    *,
    spec: CorpusSpec,
    fold: int,
    partition: str,
    label: int | None,
) -> dict[str, Any]:
    if spec.name in ("authentic_preference", "temporal_direction_control"):
        label = int(record["selected_index"])

    output = {
        "corpus_record_schema_version": TRAINING_CORPUS_SCHEMA_VERSION,
        "corpus_name": spec.name,
        "objective": spec.objective,
        "fold": fold,
        "partition": partition,
        "source_training_allowed": partition in ("train", "validation"),
        "episode_id": str(record["episode_id"]),
        "lineage_id": str(record["lineage_id"]),
        "input_text": _input_text(record),
        "candidate_a": str(record["candidate_a"]),
        "candidate_b": str(record["candidate_b"]),
        "context_before": str(record.get("context_before", "")),
        "context_after": str(record.get("context_after", "")),
        "metadata": {
            "selected_sentence_index": record.get("selected_sentence_index"),
            "sentence_position": record.get("sentence_position"),
            "edit_similarity": record.get("edit_similarity"),
            "lexical_jaccard": record.get("lexical_jaccard"),
            "v0_version_id": record.get("v0_version_id"),
            "v1_version_id": record.get("v1_version_id"),
        },
    }
    if spec.label_name is not None:
        output["label_name"] = spec.label_name
        output["label"] = label
    else:
        output["label_name"] = None
        output["label"] = None
    return output


def _input_text(record: Mapping[str, Any]) -> str:
    parts = [
        "CONTEXT_BEFORE:",
        str(record.get("context_before", "")).strip(),
        "CANDIDATE_A:",
        str(record["candidate_a"]).strip(),
        "CANDIDATE_B:",
        str(record["candidate_b"]).strip(),
        "CONTEXT_AFTER:",
        str(record.get("context_after", "")).strip(),
    ]
    return "\n".join(parts)


def _partition_stats(
    records: Sequence[Mapping[str, Any]],
    *,
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = [record.get("label") for record in records if record.get("label") is not None]
    future_revised = sum(bool(record["future_revised"]) for record in source_records)
    token_count = sum(_token_count(str(record["input_text"])) for record in records)
    return {
        "records": len(records),
        "lineages": len({str(record["lineage_id"]) for record in records}),
        "input_tokens_whitespace": token_count,
        "labelled_records": len(labels),
        "label_one_rate": sum(int(label) == 1 for label in labels) / len(labels) if labels else None,
        "future_revised": future_revised,
        "future_revised_rate": future_revised / len(source_records) if source_records else 0.0,
        "source_training_allowed": all(record["source_training_allowed"] for record in records),
    }


def _global_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    revised = sum(bool(record["future_revised"]) for record in records)
    return {
        "episodes": len(records),
        "lineages": len({str(record["lineage_id"]) for record in records}),
        "future_revised": revised,
        "future_stable": len(records) - revised,
        "future_revised_rate": revised / len(records),
    }


def _split_identity(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": split_manifest.get("seed"),
        "outer_folds": split_manifest.get("outer_folds"),
        "grouping_key": split_manifest.get("grouping_key"),
        "episodes_sha256": (
            split_manifest.get("sources", {})
            .get("episodes", {})
            .get("sha256")
        ),
        "numeric_flags_sha256": (
            split_manifest.get("sources", {})
            .get("numeric_flags", {})
            .get("sha256")
        ),
    }


def _build_gates(manifest: Mapping[str, Any]) -> dict[str, bool]:
    corpus_names = list(manifest["corpus_names"])
    folds = range(int(manifest["outer_folds"]))
    equal_records = True
    equal_tokens = True
    no_future_labels = True
    test_not_training = True
    for fold in folds:
        for partition in PARTITIONS:
            stats = [
                manifest["corpora"][name]["folds"][f"fold-{fold:02d}"][partition]
                for name in corpus_names
            ]
            equal_records = equal_records and len({item["records"] for item in stats}) == 1
            equal_tokens = equal_tokens and len(
                {item["input_tokens_whitespace"] for item in stats}
            ) == 1
            if partition == "test":
                test_not_training = test_not_training and not any(
                    item["source_training_allowed"] for item in stats
                )
    return {
        "episodes_hash_matches_frozen_split": bool(
            manifest["source_checks"]["episodes_sha256_matches_split_manifest"]
        ),
        "all_corpora_share_partition_record_counts": equal_records,
        "all_corpora_share_partition_input_token_counts": equal_tokens,
        "future_labels_redacted_from_corpus_records": no_future_labels,
        "test_partitions_marked_not_for_source_training": test_not_training,
        "all_expected_corpora_present": set(corpus_names)
        == {spec.name for spec in CORPUS_SPECS},
    }


def _build_warnings(manifest: Mapping[str, Any]) -> list[str]:
    warnings = [
        "Do not train source-task encoders on test partitions; they are emitted only for audit and optional source-task reporting.",
        "Future labels are used in the manifest summary only, not in corpus JSONL records.",
    ]
    if not all(manifest["gates"].values()):
        warnings.append("At least one Step 2 corpus gate failed; do not train from these corpora.")
    return warnings


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _source_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hex(seed: int, *parts: str) -> str:
    joined = "\0".join((str(seed), *parts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _stable_bit(seed: int, *parts: str) -> int:
    return int(_stable_hex(seed, *parts)[:2], 16) & 1


def _token_count(text: str) -> int:
    return len(text.split())
