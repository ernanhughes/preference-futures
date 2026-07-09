from __future__ import annotations

import json
from pathlib import Path

import pytest

from preference_futures.probes import contract as contract_module
from preference_futures.probes.common import L2_GRID, select_l2_candidate
from preference_futures.probes.metrics import binary_metrics
from preference_futures.probes.verify import _lineage_bootstrap
from preference_futures.selection.diagnostics import ALL_ARMS
from preference_futures.training.common import sha256_file, write_json


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_validation_selection_breaks_ties_toward_stronger_l2() -> None:
    candidates = [
        {"l2_lambda": 1e-4, "validation": {"log_loss": 0.5}},
        {"l2_lambda": 1e-3, "validation": {"log_loss": 0.5 + 1e-13}},
        {"l2_lambda": 1e-2, "validation": {"log_loss": 0.6}},
    ]
    assert select_l2_candidate(candidates)["l2_lambda"] == 1e-3


def test_binary_metrics_are_probabilistic_and_rank_aware() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["records"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] < 0.3
    assert metrics["brier_score"] < 0.03


def test_lineage_bootstrap_uses_paired_loss_difference() -> None:
    authentic = {
        "a": {"lineage_id": "x", "future_revised": 0, "probability": 0.1},
        "b": {"lineage_id": "y", "future_revised": 1, "probability": 0.9},
    }
    comparator = {
        "a": {"lineage_id": "x", "future_revised": 0, "probability": 0.4},
        "b": {"lineage_id": "y", "future_revised": 1, "probability": 0.6},
    }
    result = _lineage_bootstrap(
        authentic=authentic,
        comparator=comparator,
        seed=17,
        replicates=100,
    )
    assert result["log_loss_improvement"] > 0.0
    assert result["probability_improvement_positive"] == 1.0


def test_rejects_incomplete_step5_verification() -> None:
    representation_contract = {
        "outer_folds": 2,
        "expected_extraction_jobs": 14,
    }
    with pytest.raises(ValueError, match="has not passed"):
        contract_module._require_complete_representation_verification(
            {
                "passed": False,
                "status": "fail",
                "selection": {"folds": [0, 1], "arms": list(ALL_ARMS)},
                "observed": {"expected_jobs": 14, "observed_jobs": 13},
            },
            representation_contract,
        )


def test_builds_identical_probe_contract_from_complete_step5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    representations = tmp_path / "representations"
    representations.mkdir()
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            {"episode_id": "a", "lineage_id": "la", "future_revised": False},
            {"episode_id": "b", "lineage_id": "lb", "future_revised": True},
        ],
    )

    jobs = []
    for fold in range(2):
        for arm in ALL_ARMS:
            run_directory = representations / "runs" / f"fold-{fold:02d}" / arm
            artifacts = {}
            for partition in ("train", "validation", "test"):
                vector_path = run_directory / f"{partition}.safetensors"
                rows_path = run_directory / f"{partition}.rows.jsonl"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                vector_path.write_bytes(f"{fold}:{arm}:{partition}:vectors".encode())
                _write_jsonl(
                    rows_path,
                    [{"row_index": 0, "episode_id": "a", "lineage_id": "la"}],
                )
                artifacts[partition] = {
                    "representations_path": vector_path.name,
                    "representations_sha256": sha256_file(vector_path),
                    "rows_path": rows_path.name,
                    "rows_sha256": sha256_file(rows_path),
                    "rows": 1,
                    "shape": [1, 768],
                    "dtype": "float32",
                }
            report = {
                "status": "complete",
                "contract_sha256": "representation-contract",
                "encoder_sha256": f"encoder-{fold}-{arm}",
                "artifacts": artifacts,
            }
            write_json(run_directory / "run.json", report)
            jobs.append(
                {
                    "fold": fold,
                    "regime": arm,
                    "encoder_sha256": f"encoder-{fold}-{arm}",
                    "source_task_status": "fixture",
                }
            )

    representation_contract = {
        "contract_sha256": "representation-contract",
        "seed": 17,
        "outer_folds": 2,
        "expected_extraction_jobs": 14,
        "sources": {
            "episodes": {
                "path": str(episodes_path),
                "sha256": sha256_file(episodes_path),
            }
        },
        "jobs": jobs,
    }
    write_json(representations / "contract.json", representation_contract)
    write_json(
        representations / "representation-verification.json",
        {
            "passed": True,
            "status": "pass",
            "selection": {"folds": [0, 1], "arms": list(ALL_ARMS)},
            "observed": {"expected_jobs": 14, "observed_jobs": 14},
        },
    )
    monkeypatch.setattr(contract_module, "validate_representation_contract", lambda value: None)

    contract = contract_module.build_probe_contract(
        representation_directory=representations,
        output_directory=tmp_path / "probes",
    )

    assert contract["expected_probe_jobs"] == 14
    assert contract["probe"]["architecture"] == "single_linear_logit_with_bias"
    assert tuple(contract["probe"]["l2_grid"]) == L2_GRID
    assert contract["probe"]["training_partition"] == "train_only"
    assert contract["probe"]["selection_partition"] == "validation_only"
    assert contract["probe"]["test_evaluations_per_job"] == 1
    assert contract["confirmatory_estimand"]["bootstrap_replicates"] == 10000
    assert len(contract["jobs"]) == 14
