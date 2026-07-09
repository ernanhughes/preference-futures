"""Construct the six Step 2 source-task controls for one fold partition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from preference_futures.corpora.common import CORPUS_SCHEMA_VERSION, hash_int


def build_partition_corpora(
    episodes: Sequence[Mapping[str, Any]],
    temporal_pool: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fold: int,
    partition: str,
) -> dict[str, list[dict[str, Any]]]:
    if not episodes:
        raise ValueError(f"fold {fold} {partition} has no preference episodes")
    count = len(episodes)
    if len(temporal_pool) < count:
        raise ValueError(
            f"fold {fold} {partition} requires {count} temporal pairs but only "
            f"{len(temporal_pool)} are available"
        )

    ordered = sorted(episodes, key=lambda row: str(row["episode_id"]))
    authentic = [
        _base_record(row, "authentic_preference", fold, partition)
        | {
            "objective": "binary_pair_classification",
            "input_view": "canonical_episode_pair_with_context",
            "target": int(row["selected_index"]),
            "target_semantics": "editor_retained_candidate_index",
        }
        for row in ordered
    ]

    random_labels = _balanced_labels(
        [str(row["episode_id"]) for row in ordered], seed, fold, partition
    )
    random_label = [
        _base_record(row, "random_label", fold, partition)
        | {
            "objective": "binary_pair_classification",
            "input_view": "canonical_episode_pair_with_context",
            "target": random_labels[str(row["episode_id"])],
            "target_semantics": "deterministic_balanced_random_label",
        }
        for row in ordered
    ]

    shuffled_donors = _donor_mapping(
        ordered,
        seed=seed + 101,
        namespace=f"shuffle:{fold}:{partition}",
    )
    by_id = {str(row["episode_id"]): row for row in ordered}
    shuffled: list[dict[str, Any]] = []
    for row in ordered:
        source_id = str(row["episode_id"])
        donor_id = shuffled_donors[source_id]
        shuffled.append(
            _base_record(row, "shuffled_preference", fold, partition)
            | {
                "objective": "binary_pair_classification",
                "input_view": "canonical_episode_pair_with_context",
                "target": int(by_id[donor_id]["selected_index"]),
                "target_semantics": "preference_label_from_different_episode",
                "label_source_episode_id": donor_id,
            }
        )

    exposure_labels = _balanced_labels(
        [str(row["episode_id"]) for row in ordered],
        seed + 211,
        fold,
        partition,
    )
    exposure_donors = _donor_mapping(
        ordered,
        seed=seed + 307,
        namespace=f"pair:{fold}:{partition}",
    )
    pair_exposure: list[dict[str, Any]] = []
    for row in ordered:
        episode_id = str(row["episode_id"])
        is_true_pair = exposure_labels[episode_id] == 1
        partner = episode_id if is_true_pair else exposure_donors[episode_id]
        pair_exposure.append(
            _base_record(row, "pair_exposure", fold, partition)
            | {
                "objective": "same_revision_pair_classification",
                "input_view": "candidate_a_from_source_candidate_b_from_declared_partner",
                "target": int(is_true_pair),
                "target_semantics": "candidates_originate_from_same_revision_episode",
                "candidate_b_source_episode_id": partner,
            }
        )

    language_adaptation: list[dict[str, Any]] = []
    for row in ordered:
        tokens = serialised_episode_text(row).split()
        mask_count = max(1, min(len(tokens), int(round(len(tokens) * 0.15))))
        ranked = sorted(
            range(len(tokens)),
            key=lambda index: hash_int(
                f"mask:{seed}:{fold}:{partition}:{row['episode_id']}:{index}"
            ),
        )
        language_adaptation.append(
            _base_record(row, "language_adaptation", fold, partition)
            | {
                "objective": "deterministic_masked_word_reconstruction",
                "input_view": "serialised_episode_pair_with_context",
                "target_semantics": "reconstruct_masked_whitespace_tokens",
                "word_token_count": len(tokens),
                "mask_indices": sorted(ranked[:mask_count]),
            }
        )

    target_lengths = [episode_token_count(row) for row in ordered]
    selected_temporal = _match_length_distribution(temporal_pool, target_lengths, count)
    temporal_ids = [str(row["temporal_pair_id"]) for row in selected_temporal]
    temporal_labels = _balanced_labels(temporal_ids, seed + 401, fold, partition)
    temporal_direction = [
        {
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "corpus": "temporal_direction",
            "fold": fold,
            "partition": partition,
            "source_kind": "independent_temporal_pair",
            "source_id": str(row["temporal_pair_id"]),
            "lineage_id": str(row["lineage_id"]),
            "objective": "binary_pair_classification",
            "input_view": "orient_earlier_later_text_so_target_indexes_later_candidate",
            "target": temporal_labels[str(row["temporal_pair_id"])],
            "target_semantics": "newer_candidate_index",
        }
        for row in selected_temporal
    ]

    return {
        "language_adaptation": language_adaptation,
        "pair_exposure": pair_exposure,
        "temporal_direction": temporal_direction,
        "random_label": random_label,
        "shuffled_preference": shuffled,
        "authentic_preference": authentic,
    }


def assign_temporal_lineages(
    temporal_pairs: Sequence[Mapping[str, Any]], folds: int, seed: int
) -> dict[str, int]:
    counts = Counter(str(row["lineage_id"]) for row in temporal_pairs)
    totals = [0] * folds
    assignments: dict[str, int] = {}
    ordered = sorted(
        counts,
        key=lambda lineage: (-counts[lineage], hash_int(f"temporal-fold:{seed}:{lineage}")),
    )
    for lineage in ordered:
        fold = min(range(folds), key=lambda value: (totals[value], value))
        assignments[lineage] = fold
        totals[fold] += counts[lineage]
    return assignments


def record_exposure_tokens(
    record: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, Any]],
    temporal_pairs: Mapping[str, Mapping[str, Any]],
) -> int:
    if record["source_kind"] == "independent_temporal_pair":
        pair = temporal_pairs[str(record["source_id"])]
        return temporal_token_count(pair)
    episode = episodes[str(record["source_id"])]
    if record["corpus"] == "pair_exposure" and int(record["target"]) == 0:
        donor = episodes[str(record["candidate_b_source_episode_id"])]
        return episode_token_count(episode, candidate_b=str(donor["candidate_b"]))
    return episode_token_count(episode)


def serialised_episode_text(
    episode: Mapping[str, Any], *, candidate_b: str | None = None
) -> str:
    return "\n".join(
        (
            "[CONTEXT_BEFORE]",
            str(episode.get("context_before", "")),
            "[CANDIDATE_A]",
            str(episode["candidate_a"]),
            "[CANDIDATE_B]",
            str(episode["candidate_b"] if candidate_b is None else candidate_b),
            "[CONTEXT_AFTER]",
            str(episode.get("context_after", "")),
        )
    )


def episode_token_count(
    episode: Mapping[str, Any], *, candidate_b: str | None = None
) -> int:
    return len(serialised_episode_text(episode, candidate_b=candidate_b).split())


def temporal_token_count(pair: Mapping[str, Any]) -> int:
    return len(
        (
            f"[CONTEXT_BEFORE] {pair.get('context_before', '')} "
            f"[CANDIDATE_A] {pair['earlier_text']} "
            f"[CANDIDATE_B] {pair['later_text']} "
            f"[CONTEXT_AFTER] {pair.get('context_after', '')}"
        ).split()
    )


def _base_record(
    episode: Mapping[str, Any], corpus: str, fold: int, partition: str
) -> dict[str, Any]:
    return {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "corpus": corpus,
        "fold": fold,
        "partition": partition,
        "source_kind": "preference_episode",
        "source_id": str(episode["episode_id"]),
        "lineage_id": str(episode["lineage_id"]),
    }


def _balanced_labels(
    ids: Sequence[str], seed: int, fold: int, partition: str
) -> dict[str, int]:
    ordered = sorted(
        ids,
        key=lambda value: hash_int(f"label:{seed}:{fold}:{partition}:{value}"),
    )
    split = len(ordered) // 2
    return {value: int(index >= split) for index, value in enumerate(ordered)}


def _donor_mapping(
    records: Sequence[Mapping[str, Any]], *, seed: int, namespace: str
) -> dict[str, str]:
    if len(records) < 2:
        raise ValueError("at least two records are required for donor controls")
    ordered = sorted(
        records,
        key=lambda row: hash_int(f"{namespace}:{seed}:{row['episode_id']}"),
    )
    ids = [str(row["episode_id"]) for row in ordered]
    lineages = [str(row["lineage_id"]) for row in ordered]
    candidate_shifts = set(range(1, min(len(ids), 257)))
    candidate_shifts.update({len(ids) // 2, max(Counter(lineages).values())})
    candidate_shifts = {value % len(ids) for value in candidate_shifts if value % len(ids)}
    best_shift = min(
        candidate_shifts,
        key=lambda shift: (
            sum(
                lineages[index] == lineages[(index + shift) % len(ids)]
                for index in range(len(ids))
            ),
            sum(
                ids[index] == ids[(index + shift) % len(ids)]
                for index in range(len(ids))
            ),
            hash_int(f"{namespace}:shift:{seed}:{shift}"),
        ),
    )
    mapping = {
        ids[index]: ids[(index + best_shift) % len(ids)] for index in range(len(ids))
    }
    lineage_by_id = {
        str(row["episode_id"]): str(row["lineage_id"]) for row in ordered
    }
    if any(
        lineage_by_id[episode_id] == lineage_by_id[donor_id]
        for episode_id, donor_id in mapping.items()
    ):
        raise ValueError(f"could not construct cross-lineage donors for {namespace}")
    return mapping


def _match_length_distribution(
    pool: Sequence[Mapping[str, Any]], target_lengths: Sequence[int], count: int
) -> list[Mapping[str, Any]]:
    candidates = sorted(
        pool,
        key=lambda row: (temporal_token_count(row), str(row["temporal_pair_id"])),
    )
    if len(candidates) < count:
        raise ValueError("temporal pool is smaller than requested count")
    targets = sorted(target_lengths)
    if count == 1:
        target = targets[0]
        return [min(candidates, key=lambda row: abs(temporal_token_count(row) - target))]
    chosen_indices = {
        round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)
    }
    if len(chosen_indices) != count:
        chosen_indices = set(range(count))
    selected = [candidates[index] for index in sorted(chosen_indices)]
    selected.sort(key=lambda row: str(row["temporal_pair_id"]))
    return selected
