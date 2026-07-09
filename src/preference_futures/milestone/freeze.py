"""Build a compact, hash-addressed snapshot of the confirmatory experiment."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preference_futures.training.common import canonical_json_sha256, load_json, write_json

MILESTONE_SCHEMA_VERSION = 1
DEFAULT_MILESTONE_NAME = "preference-futures-v0.1-confirmatory-negative"
RESULT_JSON_NAME = "step-06-confirmatory-result.json"
RESULT_MARKDOWN_NAME = "step-06-confirmatory-result.md"


class MilestoneError(ValueError):
    """Raised when the confirmatory evidence chain is incomplete or inconsistent."""


def freeze_confirmatory_milestone(
    *,
    repository_root: Path,
    splits_directory: Path,
    corpora_directory: Path,
    training_directory: Path,
    encoder_selection_directory: Path,
    representation_directory: Path,
    probe_directory: Path,
    release_root: Path,
    repository_result_directory: Path,
    milestone_name: str = DEFAULT_MILESTONE_NAME,
    allow_dirty: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Validate Steps 1-6, write result records, and ZIP compact evidence."""

    repository = repository_root.expanduser().resolve()
    release_parent = release_root.expanduser().resolve()
    result_directory = repository_result_directory.expanduser().resolve()
    release_directory = release_parent / milestone_name
    archive_path = release_parent / f"{milestone_name}.zip"
    archive_hash_path = release_parent / f"{milestone_name}.zip.sha256"

    git = _git_identity(repository)
    if git["dirty"] and not allow_dirty:
        raise MilestoneError(
            "tracked repository files are dirty; commit or stash them, "
            "or pass --allow-dirty"
        )

    sources = {
        "splits": splits_directory.expanduser().resolve(),
        "corpora": corpora_directory.expanduser().resolve(),
        "training": training_directory.expanduser().resolve(),
        "encoder_selection": encoder_selection_directory.expanduser().resolve(),
        "representations": representation_directory.expanduser().resolve(),
        "probes": probe_directory.expanduser().resolve(),
    }
    _validate_evidence_chain(sources)

    result = _build_confirmatory_result(
        probe_summary=load_json(sources["probes"] / "probe-summary.json"),
        probe_verification=load_json(sources["probes"] / "probe-verification.json"),
        git=git,
        milestone_name=milestone_name,
    )

    if release_directory.exists() or archive_path.exists() or archive_hash_path.exists():
        if not force:
            raise MilestoneError(
                "milestone output already exists; pass --force to replace it: "
                f"{release_directory}"
            )
        shutil.rmtree(release_directory, ignore_errors=True)
        archive_path.unlink(missing_ok=True)
        archive_hash_path.unlink(missing_ok=True)

    release_parent.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    release_directory.mkdir(parents=True, exist_ok=True)

    result_json_path = result_directory / RESULT_JSON_NAME
    result_markdown_path = result_directory / RESULT_MARKDOWN_NAME
    write_json(result_json_path, result)
    result_markdown_path.write_text(
        render_confirmatory_result_markdown(result),
        encoding="utf-8",
    )

    copied: list[dict[str, Any]] = []
    for source_path, archive_relative in _evidence_files(sources, repository):
        target = release_directory / "evidence" / archive_relative
        _copy_and_record(source_path, target, release_directory, copied)

    for generated_path in (result_json_path, result_markdown_path):
        target = release_directory / "results" / generated_path.name
        _copy_and_record(generated_path, target, release_directory, copied)

    release_metadata = {
        "milestone_schema_version": MILESTONE_SCHEMA_VERSION,
        "milestone_name": milestone_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "result": result["primary_result"],
        "classification": result["classification"],
        "scope": {
            "confirmatory_steps": [1, 2, 3, 4, 5, 6],
            "large_encoder_checkpoints_included": False,
            "step_5_representation_matrices_included": False,
            "source_corpus_text_included": False,
            "probe_weights_included": True,
            "out_of_fold_predictions_included": True,
        },
    }
    metadata_path = release_directory / "release-metadata.json"
    readme_path = release_directory / "README.md"
    write_json(metadata_path, release_metadata)
    readme_path.write_text(
        render_release_readme(result, release_metadata),
        encoding="utf-8",
    )
    copied.extend(
        _file_record(path, release_directory, source_path=path)
        for path in (metadata_path, readme_path)
    )

    manifest = {
        "milestone_manifest_schema_version": MILESTONE_SCHEMA_VERSION,
        "milestone_name": milestone_name,
        "git_commit": git["commit"],
        "file_count": len(copied),
        "total_bytes": sum(int(item["bytes"]) for item in copied),
        "files": sorted(copied, key=lambda item: str(item["path"])),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = release_directory / "manifest.json"
    write_json(manifest_path, manifest)

    shutil.make_archive(
        str(archive_path.with_suffix("")),
        "zip",
        root_dir=release_parent,
        base_dir=milestone_name,
    )
    archive_sha256 = _sha256_file(archive_path)
    archive_hash_path.write_text(
        f"{archive_sha256}  {archive_path.name}\n",
        encoding="utf-8",
    )

    final = {
        "status": "complete",
        "milestone_name": milestone_name,
        "classification": result["classification"],
        "git": git,
        "repository_result_json": str(result_json_path),
        "repository_result_markdown": str(result_markdown_path),
        "release_directory": str(release_directory),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_hash_path": str(archive_hash_path),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "suggested_git_tag": "v0.1-confirmatory-negative",
    }
    write_json(release_parent / f"{milestone_name}.summary.json", final)
    return final


def _copy_and_record(
    source: Path,
    target: Path,
    release_directory: Path,
    records: list[dict[str, Any]],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    records.append(_file_record(target, release_directory, source_path=source))


def _validate_evidence_chain(sources: Mapping[str, Path]) -> None:
    required = {
        "splits": ["manifest.json", "split-verification.json"],
        "corpora": ["manifest.json", "corpus-verification.json"],
        "training": ["contract.json", "training-verification-confirmatory.json"],
        "encoder_selection": ["accepted-encoders.json", "source-task-summary.json"],
        "representations": ["contract.json", "representation-verification.json"],
        "probes": ["contract.json", "probe-verification.json", "probe-summary.json"],
    }
    for stage, names in required.items():
        for name in names:
            path = sources[stage] / name
            if not path.is_file():
                raise MilestoneError(f"missing {stage} evidence file: {path}")

    verification_checks = (
        (sources["splits"] / "split-verification.json", None, "split"),
        (sources["corpora"] / "corpus-verification.json", None, "corpus"),
        (
            sources["training"] / "training-verification-confirmatory.json",
            60,
            "training",
        ),
        (
            sources["representations"] / "representation-verification.json",
            70,
            "representation",
        ),
        (sources["probes"] / "probe-verification.json", 70, "probe"),
    )
    for path, expected_jobs, label in verification_checks:
        report = load_json(path)
        status_passed = str(report.get("status", "")).lower() == "pass"
        if report.get("passed") is not True and not status_passed:
            raise MilestoneError(f"{label} verification has not passed: {path}")
        if expected_jobs is not None:
            observed_jobs = report.get("observed", {}).get("observed_jobs")
            if observed_jobs is not None and int(observed_jobs) != expected_jobs:
                raise MilestoneError(
                    f"{label} verification expected {expected_jobs} jobs, "
                    f"observed {observed_jobs}"
                )

    training = load_json(sources["training"] / "training-verification-confirmatory.json")
    if training.get("mode") != "confirmatory":
        raise MilestoneError("Step 3 verification is not confirmatory")

    selection = load_json(sources["encoder_selection"] / "accepted-encoders.json")
    if selection.get("status") != "frozen_for_step_5":
        raise MilestoneError("Step 4 encoder manifest is not frozen for Step 5")
    if int(selection.get("counts", {}).get("eligible_entries", 0)) != 70:
        raise MilestoneError("Step 4 encoder manifest does not contain 70 eligible entries")

    probe_summary = load_json(sources["probes"] / "probe-summary.json")
    if probe_summary.get("status") != "complete":
        raise MilestoneError("Step 6 probe summary is incomplete")


def _build_confirmatory_result(
    *,
    probe_summary: Mapping[str, Any],
    probe_verification: Mapping[str, Any],
    git: Mapping[str, Any],
    milestone_name: str,
) -> dict[str, Any]:
    comparison = probe_summary["paired_lineage_bootstrap"][
        "authentic_preference_vs_generic"
    ]
    improvement = float(comparison["log_loss_improvement"])
    lower, upper = (float(value) for value in comparison["confidence_interval_95"])
    classification = _classify_result(lower=lower, upper=upper)

    if classification == "not_supported":
        supported_claim = (
            "Under the frozen DistilBERT, source-training, pair-and-context input, "
            "final-layer first-token representation, and identical linear-probe "
            "contract, authentic-preference training did not improve out-of-fold "
            "prediction of future revision relative to the untouched generic encoder."
        )
    elif classification == "supported":
        supported_claim = (
            "Under the frozen confirmatory contract, authentic-preference training "
            "produced a reliable out-of-fold log-loss improvement over the untouched "
            "generic encoder."
        )
    else:
        supported_claim = (
            "Under the frozen confirmatory contract, authentic-preference training "
            "reliably degraded out-of-fold prediction relative to the untouched "
            "generic encoder."
        )

    result: dict[str, Any] = {
        "confirmatory_result_schema_version": MILESTONE_SCHEMA_VERSION,
        "status": "verified",
        "milestone_name": milestone_name,
        "classification": classification,
        "git": dict(git),
        "probe_contract_sha256": probe_summary.get("contract_sha256"),
        "verification_passed": probe_verification.get("passed") is True,
        "expected_jobs": int(
            probe_verification.get("observed", {}).get("expected_jobs", 0)
        ),
        "observed_jobs": int(
            probe_verification.get("observed", {}).get("observed_jobs", 0)
        ),
        "episodes": int(probe_summary.get("episodes", 0)),
        "lineages": int(probe_summary.get("lineages", 0)),
        "primary_result": {
            "comparison": "authentic_preference_vs_generic",
            "estimand": "generic_log_loss_minus_authentic_preference_log_loss",
            "log_loss_improvement": improvement,
            "confidence_interval_95": [lower, upper],
            "positive_value_favors": "authentic_preference",
            "bootstrap_unit": "article_lineage",
            "bootstrap_replicates": int(comparison["bootstrap_replicates"]),
        },
        "arm_metrics": probe_summary.get("arm_metrics", {}),
        "supported_claim": supported_claim,
        "claim_limits": [
            "The result tests linear decodability from the frozen Step 5 representation.",
            (
                "It does not prove that preference supervision can never transfer in "
                "another model or task."
            ),
            (
                "Analyses designed after observing this result are exploratory unless "
                "replicated on fresh data."
            ),
        ],
    }
    result["result_sha256"] = canonical_json_sha256(result)
    return result


def _classify_result(*, lower: float, upper: float) -> str:
    if lower > 0.0:
        return "supported"
    if upper < 0.0:
        return "degraded"
    return "not_supported"


def _evidence_files(
    sources: Mapping[str, Path],
    repository: Path,
) -> Iterable[tuple[Path, Path]]:
    patterns = {
        "splits": [
            "manifest.json",
            "split-summary.json",
            "split-summary.md",
            "split-verification.json",
            "split-verification.md",
            "fold-*.json",
        ],
        "corpora": [
            "manifest.json",
            "corpus-summary.md",
            "corpus-verification.json",
            "corpus-verification.md",
            "temporal-pairs-audit.json",
        ],
        "training": [
            "contract.json",
            "training-plan.md",
            "training-verification-confirmatory.json",
            "training-verification-confirmatory.md",
            "runs/**/run.json",
            "runs/**/metrics.jsonl",
        ],
        "encoder_selection": ["*.json", "*.md"],
        "representations": [
            "contract.json",
            "extraction-plan.md",
            "representation-verification.json",
            "representation-verification.md",
            "runs/**/run.json",
            "runs/**/*.rows.jsonl",
        ],
        "probes": [
            "contract.json",
            "probe-plan.md",
            "probe-verification.json",
            "probe-verification.md",
            "probe-summary.json",
            "probe-summary.md",
            "runs/**/run.json",
            "runs/**/validation.predictions.jsonl",
            "runs/**/test.predictions.jsonl",
            "runs/**/probe.safetensors",
        ],
    }
    seen: set[Path] = set()
    for stage, stage_patterns in patterns.items():
        root = sources[stage]
        for pattern in stage_patterns:
            for path in sorted(root.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    yield path, Path(stage) / path.relative_to(root)

    docs_results = repository / "docs" / "results"
    generated_names = {RESULT_JSON_NAME, RESULT_MARKDOWN_NAME}
    if docs_results.is_dir():
        for path in sorted(docs_results.glob("step-*")):
            if path.is_file() and path.name not in generated_names and path not in seen:
                seen.add(path)
                yield path, Path("repository-results") / path.name


def _file_record(path: Path, root: Path, *, source_path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "source_path": str(source_path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current") or "detached"
        status_lines = [
            line for line in run("status", "--porcelain").splitlines() if line
        ]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MilestoneError(f"unable to inspect Git repository: {repository}") from exc
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_lines),
        "dirty_entries": status_lines,
    }


def render_confirmatory_result_markdown(result: Mapping[str, Any]) -> str:
    primary = result["primary_result"]
    lower, upper = primary["confidence_interval_95"]
    status = str(result["classification"]).replace("_", " ").upper()
    lines = [
        "# Step 6 Confirmatory Result",
        "",
        f"**Status:** VERIFIED — {status}",
        "",
        "## Primary comparison",
        "",
        "```text",
        "generic pooled out-of-fold log loss",
        "-",
        "authentic-preference pooled out-of-fold log loss",
        "```",
        "",
        f"**Observed improvement:** {primary['log_loss_improvement']:+.6f}",
        "",
        f"**Paired lineage-bootstrap 95% CI:** [{lower:+.6f}, {upper:+.6f}]",
        "",
        f"**Episodes:** {result['episodes']:,}",
        "",
        f"**Lineages:** {result['lineages']:,}",
        "",
        f"**Probe jobs:** {result['observed_jobs']}/{result['expected_jobs']} verified",
        "",
        "## Conclusion",
        "",
        str(result["supported_claim"]),
        "",
        "## Claim limits",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["claim_limits"])
    lines.extend(["", "## Seven-arm results", ""])
    lines.append("| Arm | Log loss | Improvement vs generic | Brier | ROC AUC |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm, metrics in result.get("arm_metrics", {}).items():
        auc = metrics.get("roc_auc")
        auc_text = "—" if auc is None else f"{float(auc):.6f}"
        lines.append(
            f"| {arm} | {float(metrics['log_loss']):.6f} | "
            f"{float(metrics.get('log_loss_improvement_vs_generic', 0.0)):+.6f} | "
            f"{float(metrics['brier_score']):.6f} | {auc_text} |"
        )
    lines.extend(
        [
            "",
            "This file marks the completed confirmatory experiment. Subsequent analyses are",
            "exploratory unless repeated on fresh held-out data.",
            "",
        ]
    )
    return "\n".join(lines)


def render_release_readme(
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    primary = result["primary_result"]
    lower, upper = primary["confidence_interval_95"]
    classification = str(result["classification"]).replace("_", " ")
    return "\n".join(
        [
            f"# {metadata['milestone_name']}",
            "",
            "This archive freezes the compact evidence chain for the completed Steps 1-6",
            "confirmatory experiment.",
            "",
            "## Headline result",
            "",
            f"- Classification: **{classification}**",
            (
                "- Authentic-versus-generic log-loss improvement: "
                f"`{primary['log_loss_improvement']:+.6f}`"
            ),
            f"- Paired lineage-bootstrap 95% CI: `[{lower:+.6f}, {upper:+.6f}]`",
            f"- Verified probe jobs: `{result['observed_jobs']}/{result['expected_jobs']}`",
            f"- Git commit before snapshot generation: `{metadata['git']['commit']}`",
            "",
            "## Included",
            "",
            "- frozen contracts, manifests, verification reports, and summaries for Steps 1-6;",
            "- Step 3 run reports and metric trajectories, without encoder checkpoints;",
            "- Step 5 row identities and run reports, without representation matrices;",
            "- Step 6 probe weights, run reports, and out-of-fold predictions;",
            "- a SHA-256 manifest covering every file in this directory.",
            "",
            "## Excluded",
            "",
            "- original NewsEdits sentence text and source-task corpora;",
            "- trained Transformer checkpoints;",
            "- frozen Step 5 representation matrices.",
            "",
            "The exclusions keep the milestone compact and avoid redistributing source text.",
            "Large local artifacts remain verifiable through the persisted hashes.",
            "",
        ]
    )
