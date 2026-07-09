from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from preference_futures.milestone.freeze import (
    MilestoneError,
    freeze_confirmatory_milestone,
)
from preference_futures.training.common import write_json


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _fixture_repository(tmp_path: Path) -> dict[str, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    directories = {
        "repository": repository,
        "splits": repository / "artifacts" / "transfer" / "splits",
        "corpora": repository / "artifacts" / "transfer" / "corpora",
        "training": repository / "artifacts" / "transfer" / "training",
        "encoder_selection": repository / "artifacts" / "transfer" / "encoder-selection",
        "representations": repository / "artifacts" / "transfer" / "representations",
        "probes": repository / "artifacts" / "transfer" / "probes",
        "releases": repository / "artifacts" / "releases",
        "results": repository / "docs" / "results",
    }
    for directory in directories.values():
        if directory != repository:
            directory.mkdir(parents=True, exist_ok=True)

    write_json(directories["splits"] / "manifest.json", {"outer_folds": 10})
    write_json(directories["splits"] / "split-verification.json", {"passed": True})
    write_json(directories["corpora"] / "manifest.json", {"status": "complete"})
    write_json(directories["corpora"] / "corpus-verification.json", {"passed": True})
    write_json(directories["training"] / "contract.json", {"contract_sha256": "train"})
    write_json(
        directories["training"] / "training-verification-confirmatory.json",
        {
            "passed": True,
            "mode": "confirmatory",
            "observed": {"observed_jobs": 60},
        },
    )
    write_json(
        directories["encoder_selection"] / "accepted-encoders.json",
        {
            "status": "frozen_for_step_5",
            "counts": {"eligible_entries": 70},
        },
    )
    write_json(
        directories["encoder_selection"] / "source-task-summary.json",
        {"status": "complete"},
    )
    write_json(
        directories["representations"] / "contract.json",
        {"contract_sha256": "representations"},
    )
    write_json(
        directories["representations"] / "representation-verification.json",
        {
            "passed": True,
            "status": "pass",
            "observed": {"observed_jobs": 70},
        },
    )
    write_json(directories["probes"] / "contract.json", {"contract_sha256": "probes"})
    write_json(
        directories["probes"] / "probe-verification.json",
        {
            "passed": True,
            "status": "pass",
            "observed": {"expected_jobs": 70, "observed_jobs": 70},
        },
    )
    arm_metrics = {
        arm: {
            "log_loss": 0.69,
            "log_loss_improvement_vs_generic": 0.0,
            "brier_score": 0.249,
            "roc_auc": 0.51,
        }
        for arm in (
            "generic",
            "language_adaptation",
            "pair_exposure",
            "temporal_direction",
            "random_label",
            "shuffled_preference",
            "authentic_preference",
        )
    }
    write_json(
        directories["probes"] / "probe-summary.json",
        {
            "status": "complete",
            "contract_sha256": "probes",
            "episodes": 12056,
            "lineages": 3386,
            "arm_metrics": arm_metrics,
            "paired_lineage_bootstrap": {
                "authentic_preference_vs_generic": {
                    "log_loss_improvement": -0.000719,
                    "confidence_interval_95": [-0.003425, 0.002020],
                    "bootstrap_replicates": 10000,
                }
            },
        },
    )
    probe_run = directories["probes"] / "runs" / "fold-00" / "generic"
    probe_run.mkdir(parents=True)
    write_json(probe_run / "run.json", {"status": "complete"})
    _write_jsonl(
        probe_run / "test.predictions.jsonl",
        [{"episode_id": "episode-1", "future_revised": False, "probability": 0.2}],
    )
    _write_jsonl(
        probe_run / "validation.predictions.jsonl",
        [{"episode_id": "episode-2", "future_revised": True, "probability": 0.8}],
    )
    (probe_run / "probe.safetensors").write_bytes(b"probe")
    representation_run = directories["representations"] / "runs" / "fold-00" / "generic"
    representation_run.mkdir(parents=True)
    write_json(representation_run / "run.json", {"status": "complete"})
    _write_jsonl(
        representation_run / "test.rows.jsonl",
        [{"episode_id": "episode-1", "input_sha256": "abc"}],
    )
    (representation_run / "test.safetensors").write_bytes(b"large-matrix-not-included")

    (repository / ".gitignore").write_text("artifacts/releases/\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    return directories


def test_freezes_compact_negative_milestone(tmp_path: Path) -> None:
    paths = _fixture_repository(tmp_path)

    summary = freeze_confirmatory_milestone(
        repository_root=paths["repository"],
        splits_directory=paths["splits"],
        corpora_directory=paths["corpora"],
        training_directory=paths["training"],
        encoder_selection_directory=paths["encoder_selection"],
        representation_directory=paths["representations"],
        probe_directory=paths["probes"],
        release_root=paths["releases"],
        repository_result_directory=paths["results"],
    )

    assert summary["classification"] == "not_supported"
    assert Path(summary["archive_path"]).is_file()
    assert Path(summary["archive_hash_path"]).is_file()
    assert (paths["results"] / "step-06-confirmatory-result.json").is_file()
    with zipfile.ZipFile(summary["archive_path"]) as archive:
        names = set(archive.namelist())
    assert any(name.endswith("probe.safetensors") for name in names)
    assert not any(name.endswith("representations/test.safetensors") for name in names)
    assert not any("episodes.jsonl" in name for name in names)


def test_rejects_dirty_tracked_repository(tmp_path: Path) -> None:
    paths = _fixture_repository(tmp_path)
    (paths["repository"] / ".gitignore").write_text("changed\n", encoding="utf-8")

    with pytest.raises(MilestoneError, match="dirty"):
        freeze_confirmatory_milestone(
            repository_root=paths["repository"],
            splits_directory=paths["splits"],
            corpora_directory=paths["corpora"],
            training_directory=paths["training"],
            encoder_selection_directory=paths["encoder_selection"],
            representation_directory=paths["representations"],
            probe_directory=paths["probes"],
            release_root=paths["releases"],
            repository_result_directory=paths["results"],
        )
