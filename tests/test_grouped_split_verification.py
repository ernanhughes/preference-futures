from __future__ import annotations

from copy import deepcopy

from preference_futures.splits import build_grouped_split_manifest
from preference_futures.splits.verify import verify_grouped_split_manifest


def _balanced_fixture() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    flags: dict[str, dict[str, object]] = {}
    for lineage_index in range(100):
        lineage_id = f"nyt::{lineage_index:03d}"
        for episode_index in range(2):
            episode_id = f"{lineage_id}::{episode_index}"
            records.append(
                {
                    "episode_id": episode_id,
                    "lineage_id": lineage_id,
                    "future_revised": episode_index == 0,
                    "selected_index": episode_index,
                }
            )
            flags[episode_id] = {
                "episode_id": episode_id,
                "number_changed": episode_index == 0,
                "number_dominant_edit": False,
                "casualty_count_update": False,
            }
    return records, flags


def test_verifies_complete_grouped_manifest() -> None:
    records, flags = _balanced_fixture()
    manifest, _ = build_grouped_split_manifest(
        records,
        numeric_flags=flags,
        folds=10,
        seed=17,
    )

    report = verify_grouped_split_manifest(manifest)

    assert report["passed"] is True
    assert report["errors"] == []
    assert all(report["checks"].values())
    assert report["observed"]["assignment_count"] == 100


def test_rejects_manifest_with_missing_lineage_assignment() -> None:
    records, flags = _balanced_fixture()
    manifest, _ = build_grouped_split_manifest(
        records,
        numeric_flags=flags,
        folds=10,
        seed=17,
    )
    broken = deepcopy(manifest)
    broken["lineage_to_outer_fold"].pop(next(iter(broken["lineage_to_outer_fold"])))

    report = verify_grouped_split_manifest(broken)

    assert report["passed"] is False
    assert report["checks"]["assignment_count_matches_total_lineages"] is False
    assert any("size does not match" in error for error in report["errors"])
