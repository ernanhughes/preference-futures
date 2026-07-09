"""Deterministic descriptive audit for extracted preference-futures episodes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONTEXT_AUDIT_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = {
    "schema_version",
    "episode_id",
    "lineage_id",
    "candidate_a",
    "candidate_b",
    "selected_index",
    "future_revised",
    "v0_version_id",
    "v1_version_id",
    "v2_version_id",
    "selected_sentence_index",
    "context_before",
    "context_after",
    "sentence_position",
    "edit_similarity",
    "lexical_jaccard",
}
_HONORIFIC_ONLY = re.compile(
    r'^[“"(\[]?(?:Mr|Mrs|Ms|Dr|Gen|Gov|Col|Lt|Capt|Adm|Sgt|Cpl|Sen|Rep|Prof|St)\.$',
    re.IGNORECASE,
)
_BOILERPLATE_TERMS = (
    "contributed reporting",
    "this article has been revised",
    "correction:",
    "sign up for",
    "newsletter",
    "[the new york times]",
    "_____",
)


def load_episode_records(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate a JSONL episode artifact."""

    records: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"line {line_number} must contain a JSON object")
            _validate_record(record, line_number=line_number)
            records.append(record)
    if not records:
        raise ValueError(f"no episodes found in {path}")
    return records


def build_context_viability_report(
    records: Sequence[Mapping[str, Any]],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Build the frozen descriptive audit used before grouped model experiments."""

    if not records:
        raise ValueError("records must not be empty")

    episodes = len(records)
    revised = sum(bool(record["future_revised"]) for record in records)
    selected_counts = Counter(int(record["selected_index"]) for record in records)
    lineage_counts = Counter(str(record["lineage_id"]) for record in records)
    both_context = sum(
        bool(str(record["context_before"]).strip())
        and bool(str(record["context_after"]).strip())
        for record in records
    )
    boundary_artifacts = sum(_has_boundary_artifact(record) for record in records)
    boilerplate = sum(_has_boilerplate(record) for record in records)

    report: dict[str, Any] = {
        "audit_schema_version": CONTEXT_AUDIT_SCHEMA_VERSION,
        "source": _source_metadata(source_path),
        "dataset": {
            "episodes": episodes,
            "lineages": len(lineage_counts),
            "schema_versions": sorted({int(record["schema_version"]) for record in records}),
        },
        "target_balance": {
            "future_revised": revised,
            "future_stable": episodes - revised,
            "future_revised_rate": revised / episodes,
            "future_revised_wilson_95": _wilson_interval(revised, episodes),
        },
        "candidate_orientation": {
            "selected_a": selected_counts[0],
            "selected_b": selected_counts[1],
            "selected_b_rate": selected_counts[1] / episodes,
            "selected_b_wilson_95": _wilson_interval(selected_counts[1], episodes),
        },
        "lineages": _lineage_summary(lineage_counts, episodes=episodes),
        "context": {
            "context_before_empty": sum(
                not str(record["context_before"]).strip() for record in records
            ),
            "context_after_empty": sum(
                not str(record["context_after"]).strip() for record in records
            ),
            "both_context_sides_present": both_context,
            "both_context_sides_present_rate": both_context / episodes,
            "source_boundary_artifacts": boundary_artifacts,
            "source_boundary_artifact_rate": boundary_artifacts / episodes,
            "boilerplate_or_credit_records": boilerplate,
            "boilerplate_or_credit_rate": boilerplate / episodes,
        },
        "pair_reuse": _pair_summary(records),
        "feature_distributions": {
            "edit_similarity": _distribution(
                [float(record["edit_similarity"]) for record in records]
            ),
            "lexical_jaccard": _distribution(
                [float(record["lexical_jaccard"]) for record in records]
            ),
            "sentence_position": _distribution(
                [float(record["sentence_position"]) for record in records]
            ),
            "edit_similarity_bands": _band_summary(
                records,
                field="edit_similarity",
                boundaries=(0.15, 0.4, 0.6, 0.8, 0.9, 0.95, 0.97, 0.980000001),
            ),
            "sentence_position_bands": _band_summary(
                records,
                field="sentence_position",
                boundaries=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.000000001),
            ),
            "v1_version_bands": _version_summary(records),
        },
    }
    report["gates"] = _build_gates(report)
    report["warnings"] = _build_warnings(report)
    return report


def render_context_viability_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the JSON audit."""

    dataset = report["dataset"]
    target = report["target_balance"]
    orientation = report["candidate_orientation"]
    lineages = report["lineages"]
    context = report["context"]
    pairs = report["pair_reuse"]
    lines = [
        "# Context Viability Audit",
        "",
        "## Dataset",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Episodes | {dataset['episodes']:,} |",
        f"| Article lineages | {dataset['lineages']:,} |",
        f"| Revised futures | {target['future_revised']:,} |",
        f"| Stable futures | {target['future_stable']:,} |",
        f"| Future-revision rate | {target['future_revised_rate']:.4f} |",
        f"| Selected candidate B rate | {orientation['selected_b_rate']:.4f} |",
        "",
        "## Lineage structure",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Mean episodes per lineage | {lineages['mean_episodes']:.3f} |",
        f"| Median episodes per lineage | {lineages['median_episodes']:.3f} |",
        f"| Maximum episodes in one lineage | {lineages['max_episodes']:,} |",
        f"| Episodes in top 1% of lineages | {lineages['top_1_percent_episode_rate']:.4f} |",
        f"| Episodes in top 10% of lineages | {lineages['top_10_percent_episode_rate']:.4f} |",
        "",
        "## Context and artifact flags",
        "",
        "| Measure | Count | Rate |",
        "|---|---:|---:|",
        _count_rate_row(
            "Both context sides present",
            context["both_context_sides_present"],
            context["both_context_sides_present_rate"],
        ),
        _count_rate_row(
            "Source-boundary artifacts",
            context["source_boundary_artifacts"],
            context["source_boundary_artifact_rate"],
        ),
        _count_rate_row(
            "Boilerplate or credit records",
            context["boilerplate_or_credit_records"],
            context["boilerplate_or_credit_rate"],
        ),
        "",
        "## Exact pair reuse",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Unique unordered candidate pairs | {pairs['unique_pairs']:,} |",
        f"| Repeated pair groups | {pairs['repeated_pair_groups']:,} |",
        f"| Reversal pair groups | {pairs['reversal_pair_groups']:,} |",
        f"| Episodes in reversal groups | {pairs['episodes_in_reversal_groups']:,} |",
        f"| Reversal-episode rate | {pairs['reversal_episode_rate']:.4f} |",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Experimental consequence",
            "",
            "All train, validation, test and bootstrap operations must group by `lineage_id`.",
            "Version identifiers, sentence position, edit similarity, lexical overlap and artifact",
            "flags must be retained as explicit shortcut baselines rather than being supplied only",
            "to learned text systems.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_record(record: Mapping[str, Any], *, line_number: int) -> None:
    missing = sorted(_REQUIRED_FIELDS.difference(record))
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"line {line_number} is missing required fields: {fields}")
    if type(record["selected_index"]) is not int or record["selected_index"] not in (0, 1):
        raise ValueError(f"line {line_number} has invalid selected_index")
    if type(record["future_revised"]) is not bool:
        raise TypeError(f"line {line_number} future_revised must be a bool")


def _source_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _normalise_text(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def _selected_text(record: Mapping[str, Any]) -> str:
    key = "candidate_a" if int(record["selected_index"]) == 0 else "candidate_b"
    return _normalise_text(record[key])


def _pair_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        pair = tuple(
            sorted(
                (_normalise_text(record["candidate_a"]), _normalise_text(record["candidate_b"]))
            )
        )
        groups[pair].append(record)

    repeated = [values for values in groups.values() if len(values) > 1]
    reversals = [
        values
        for values in repeated
        if len({_selected_text(record) for record in values}) > 1
    ]
    same_lineage = sum(
        len({str(record["lineage_id"]) for record in values}) == 1 for values in reversals
    )
    reversal_episodes = sum(len(values) for values in reversals)
    return {
        "unique_pairs": len(groups),
        "repeated_pair_groups": len(repeated),
        "reversal_pair_groups": len(reversals),
        "same_lineage_reversal_groups": same_lineage,
        "cross_lineage_reversal_groups": len(reversals) - same_lineage,
        "episodes_in_reversal_groups": reversal_episodes,
        "reversal_episode_rate": reversal_episodes / len(records),
    }


def _lineage_summary(counts: Counter[str], *, episodes: int) -> dict[str, Any]:
    values = sorted(counts.values(), reverse=True)

    def top_rate(fraction: float) -> float:
        count = max(1, math.ceil(len(values) * fraction))
        return sum(values[:count]) / episodes

    return {
        "mean_episodes": statistics.mean(values),
        "median_episodes": statistics.median(values),
        "max_episodes": max(values),
        "single_episode_lineages": sum(value == 1 for value in values),
        "top_1_percent_episode_rate": top_rate(0.01),
        "top_5_percent_episode_rate": top_rate(0.05),
        "top_10_percent_episode_rate": top_rate(0.10),
        "top_20_percent_episode_rate": top_rate(0.20),
        "episode_count_quantiles": _quantiles(values),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "quantiles": _quantiles(values),
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    probabilities = {
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    return {label: _quantile(ordered, probability) for label, probability in probabilities.items()}


def _quantile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _band_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    bands: list[dict[str, Any]] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = [record for record in records if lower <= float(record[field]) < upper]
        revised = sum(bool(record["future_revised"]) for record in selected)
        bands.append(
            {
                "lower": lower,
                "upper": upper,
                "episodes": len(selected),
                "future_revised": revised,
                "future_revised_rate": revised / len(selected) if selected else None,
            }
        )
    return bands


def _version_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        version = _numeric_version(record["v1_version_id"])
        if version is None:
            label = "non_numeric"
        elif version <= 5:
            label = str(int(version))
        elif version <= 10:
            label = "6-10"
        else:
            label = "11+"
        buckets[label].append(record)

    labels = ("1", "2", "3", "4", "5", "6-10", "11+", "non_numeric")
    result = []
    for label in labels:
        values = buckets.get(label, [])
        if not values:
            continue
        revised = sum(bool(record["future_revised"]) for record in values)
        result.append(
            {
                "v1_version_band": label,
                "episodes": len(values),
                "future_revised": revised,
                "future_revised_rate": revised / len(values),
            }
        )
    return result


def _numeric_version(value: object) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def _has_boundary_artifact(record: Mapping[str, Any]) -> bool:
    before = str(record["context_before"]).strip()
    after = str(record["context_after"]).strip()
    if _HONORIFIC_ONLY.match(before) or _HONORIFIC_ONLY.match(after):
        return True
    if before in {".", ";", "&amp;"} or after in {".", ";", "&amp;"}:
        return True
    for key in ("candidate_a", "candidate_b"):
        words = str(record[key]).strip().split()
        if words and _HONORIFIC_ONLY.match(words[-1]):
            return True
    return False


def _has_boilerplate(record: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(record[key])
        for key in ("candidate_a", "candidate_b", "context_before", "context_after")
    ).casefold()
    return any(term in text for term in _BOILERPLATE_TERMS)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"lower": center - half_width, "upper": center + half_width}


def _build_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    dataset = report["dataset"]
    target = report["target_balance"]
    orientation = report["candidate_orientation"]
    context = report["context"]
    return {
        "at_least_10000_episodes": int(dataset["episodes"]) >= 10_000,
        "at_least_3000_lineages": int(dataset["lineages"]) >= 3_000,
        "usable_target_balance": 0.15 <= float(target["future_revised_rate"]) <= 0.40,
        "candidate_orientation_balanced": 0.45 <= float(orientation["selected_b_rate"]) <= 0.55,
        "source_boundary_artifacts_below_3_percent": (
            float(context["source_boundary_artifact_rate"]) <= 0.03
        ),
    }


def _build_warnings(report: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    version_bands = report["feature_distributions"]["v1_version_bands"]
    if version_bands:
        rates = [float(band["future_revised_rate"]) for band in version_bands]
        if max(rates) - min(rates) >= 0.05:
            warnings.append(
                "Future-revision rates vary by V1 version band; version identifiers require an "
                "explicit temporal-position shortcut baseline."
            )
    if float(report["pair_reuse"]["reversal_episode_rate"]) >= 0.01:
        warnings.append(
            "Exact candidate-pair reversals are material; keep them grouped by article lineage and "
            "report an ablation excluding reversal groups."
        )
    if float(report["context"]["source_boundary_artifact_rate"]) > 0:
        warnings.append(
            "The official source contains residual boundary artifacts; retain a clean-prose flag and "
            "report all-data versus clean-prose results."
        )
    warnings.append("Run a grouped metadata-only future probe before text representation training.")
    return warnings


def _count_rate_row(label: str, count: object, rate: object) -> str:
    return f"| {label} | {int(count):,} | {float(rate):.4f} |"
