from __future__ import annotations

import json
from pathlib import Path

from preference_futures.editorial_mrq.oracle import (
    export_swapped_oracle_prompts,
    score_swapped_oracle_predictions,
)
from preference_futures.editorial_mrq.runtime import partition_row_indices


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_oracle_swap_exchanges_candidates_without_answer_key(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_jsonl(
        prompts,
        [
            {
                "item_id": 7,
                "episode_id": "episode-7",
                "instruction": "Return A or B.",
                "context_before": "Before.",
                "candidate_a": "Candidate alpha.",
                "candidate_b": "Candidate beta.",
                "context_after": "After.",
            }
        ],
    )

    manifest = export_swapped_oracle_prompts(
        prompts_path=prompts,
        output_directory=tmp_path / "swapped",
    )
    swapped = [
        json.loads(line)
        for line in (tmp_path / "swapped" / "oracle-prompts-swapped.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["answer_key_opened"] is False
    assert swapped[0]["candidate_a"] == "Candidate beta."
    assert swapped[0]["candidate_b"] == "Candidate alpha."
    assert swapped[0]["episode_id"] == "episode-7"


def test_swapped_oracle_scoring_translates_orientation(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    swapped = tmp_path / "swapped.jsonl"
    answers = tmp_path / "answers.jsonl"
    _write_jsonl(
        original,
        [
            {"item_id": 0, "prediction": "A"},
            {"item_id": 1, "prediction": "A"},
        ],
    )
    _write_jsonl(
        swapped,
        [
            {"item_id": 0, "prediction": "B"},
            {"item_id": 1, "prediction": "A"},
        ],
    )
    _write_jsonl(
        answers,
        [
            {"item_id": 0, "selected": "A", "selected_index": 0},
            {"item_id": 1, "selected": "B", "selected_index": 1},
        ],
    )

    report = score_swapped_oracle_predictions(
        original_predictions_path=original,
        swapped_predictions_path=swapped,
        answer_key_path=answers,
        output_directory=tmp_path / "score",
    )

    assert report["original"]["accuracy"] == 0.5
    assert report["swapped_translated"]["accuracy"] == 1.0
    assert report["order_consistency"]["rate"] == 0.5
    assert report["consistent_subset"]["accuracy"] == 1.0


def test_partition_rows_reuses_original_lineage_policy() -> None:
    rows = [
        {"lineage_id": "lineage-0"},
        {"lineage_id": "lineage-1"},
        {"lineage_id": "lineage-2"},
        {"lineage_id": "lineage-3"},
    ]
    assignments = {
        "lineage-0": 0,
        "lineage-1": 1,
        "lineage-2": 2,
        "lineage-3": 3,
    }

    partitions = partition_row_indices(
        rows,
        assignments,
        fold=0,
        outer_folds=4,
    )

    assert partitions["test"] == [0]
    assert partitions["validation"] == [1]
    assert partitions["train"] == [2, 3]
