from __future__ import annotations

import json
import math
from pathlib import Path

from preference_futures.selection import diagnostics
from preference_futures.training.common import (
    TRAINED_REGIMES,
    sha256_directory,
    write_json,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_freezes_all_seven_arms_without_using_source_success_as_gate(
    tmp_path: Path, monkeypatch
) -> None:
    training = tmp_path / "training"
    output = tmp_path / "selection"
    base_encoder = training / "base-snapshot" / "encoder"
    base_encoder.mkdir(parents=True)
    (base_encoder / "model.safetensors").write_bytes(b"base")

    jobs = []
    for fold in range(2):
        for regime in TRAINED_REGIMES:
            validation_path = (
                tmp_path / "corpora" / f"fold-{fold:02d}" / regime / "validation.jsonl"
            )
            records = [
                {"target": index % 2, "source_id": f"{fold}-{regime}-{index}"}
                for index in range(10)
            ]
            _write_jsonl(validation_path, records)
            jobs.append(
                {
                    "fold": fold,
                    "regime": regime,
                    "validation": {"path": str(validation_path), "records": 10},
                }
            )

    contract = {
        "contract_sha256": "fixture-contract",
        "outer_folds": 2,
        "trained_regimes": list(TRAINED_REGIMES),
        "expected_training_jobs": 12,
        "model": {"base_snapshot_path": str(training / "base-snapshot")},
        "optimisation": {"update_steps": 4},
        "jobs": jobs,
    }
    training.mkdir(parents=True, exist_ok=True)
    write_json(training / "contract.json", contract)
    write_json(
        training / "training-verification-confirmatory.json",
        {
            "passed": True,
            "mode": "confirmatory",
            "selection": {"folds": [0, 1], "regimes": list(TRAINED_REGIMES)},
            "observed": {"expected_jobs": 12, "observed_jobs": 12},
        },
    )
    monkeypatch.setattr(diagnostics, "validate_training_contract", lambda value: None)

    for fold in range(2):
        for regime in TRAINED_REGIMES:
            run_directory = training / "runs" / f"fold-{fold:02d}" / regime
            encoder = run_directory / "encoder"
            encoder.mkdir(parents=True)
            (encoder / "model.safetensors").write_bytes(f"{fold}:{regime}".encode())
            encoder_hash = sha256_directory(encoder)
            if regime == "language_adaptation":
                validation = {
                    "accuracy": 0.4,
                    "mean_loss": 3.5,
                    "perplexity": math.exp(3.5),
                    "records": 10,
                    "supervised_units": 100,
                    "mask_fallback_examples": 0,
                }
            elif regime == "pair_exposure":
                validation = {"accuracy": 0.9, "mean_loss": 0.2, "records": 10}
            else:
                validation = {
                    "accuracy": 0.5,
                    "mean_loss": math.log(2.0),
                    "records": 10,
                }
            write_json(
                run_directory / "run.json",
                {
                    "status": "complete",
                    "non_confirmatory": False,
                    "contract_sha256": "fixture-contract",
                    "fold": fold,
                    "regime": regime,
                    "validation": validation,
                    "artifacts": {"encoder_sha256": encoder_hash},
                },
            )
            _write_jsonl(
                run_directory / "metrics.jsonl",
                [
                    {"step": 2, "mean_training_loss_since_last_log": 1.0},
                    {"step": 4, "mean_training_loss_since_last_log": 0.5},
                ],
            )

    manifest = diagnostics.freeze_encoder_selection(training, output)

    assert manifest["status"] == "frozen_for_step_5"
    assert manifest["counts"] == {
        "folds": 2,
        "arms_per_fold": 7,
        "entries": 14,
        "eligible_entries": 14,
    }
    authentic = [
        entry for entry in manifest["entries"] if entry["regime"] == "authentic_preference"
    ]
    assert all(entry["source_task_status"] == "null_like" for entry in authentic)
    assert all(entry["eligible_for_downstream"] for entry in authentic)
    pair = [entry for entry in manifest["entries"] if entry["regime"] == "pair_exposure"]
    assert all(entry["source_task_status"] == "learned_above_prior" for entry in pair)
    assert len([entry for entry in manifest["entries"] if entry["regime"] == "generic"]) == 2
    assert json.loads((output / "encoder-hash-audit.json").read_text())["passed"] is True


def test_wilson_interval_contains_half_at_chance() -> None:
    lower, upper = diagnostics._wilson_interval(50, 100)
    assert lower < 0.5 < upper
