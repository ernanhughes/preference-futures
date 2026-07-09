"""Numeric-volatility audit for revision-derived future-prediction episodes."""

from __future__ import annotations

import difflib
import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

NUMERIC_AUDIT_SCHEMA_VERSION = 1

_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:[$£€]\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|(?:st|nd|rd|th)\b)?",
    re.IGNORECASE,
)
_MONTH_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.IGNORECASE,
)
_CASUALTY_PATTERN = re.compile(
    r"\b(?:dead|deaths?|died|killed|injured|wounded|fatalities|casualties|"
    r"death\s+toll|missing)\b",
    re.IGNORECASE,
)
_SPORTS_PATTERN = re.compile(
    r"\b(?:game|games|match|matches|season|inning|innings|quarter|halftime|"
    r"touchdown|goal|goals|runs?|points?|score|scored|defeated|beat|won|lost|"
    r"league|tournament|playoffs?|series)\b",
    re.IGNORECASE,
)
_MONEY_PATTERN = re.compile(
    r"(?:[$£€]\s*\d)|(?:\b\d+(?:\.\d+)?\s*(?:dollars?|euros?|pounds?|"
    r"million|billion|trillion)\b)",
    re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(r"(?:\d+(?:\.\d+)?\s*%)|(?:\bpercent(?:age)?\b)", re.IGNORECASE)
_UPDATE_PATTERN = re.compile(r"\b(?:updated?|update|as\s+of|revised?)\b", re.IGNORECASE)


def build_numeric_shortcut_report(
    records: Sequence[Mapping[str, Any]],
    *,
    source_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate numeric-shortcut evidence and per-episode flags."""

    if not records:
        raise ValueError("records must not be empty")

    flagged = [classify_numeric_episode(record) for record in records]
    total = len(flagged)
    total_revised = sum(bool(item["future_revised"]) for item in flagged)

    category_names = (
        "contains_number",
        "number_changed",
        "number_only_edit",
        "number_dominant_edit",
        "date_or_update_edit",
        "money_or_percentage_edit",
        "sports_numeric_edit",
        "casualty_count_update",
    )
    categories = {
        name: _category_summary(flagged, name=name, total=total, total_revised=total_revised)
        for name in category_names
    }

    lineage_numeric_counts: Counter[str] = Counter()
    lineage_casualty_counts: Counter[str] = Counter()
    for item in flagged:
        lineage = str(item["lineage_id"])
        if item["number_changed"]:
            lineage_numeric_counts[lineage] += 1
        if item["casualty_count_update"]:
            lineage_casualty_counts[lineage] += 1

    repeated_numeric_lineages = {
        lineage for lineage, count in lineage_numeric_counts.items() if count >= 2
    }
    repeated_casualty_lineages = {
        lineage for lineage, count in lineage_casualty_counts.items() if count >= 2
    }
    for item in flagged:
        lineage = str(item["lineage_id"])
        item["repeated_numeric_trajectory"] = lineage in repeated_numeric_lineages
        item["repeated_casualty_trajectory"] = lineage in repeated_casualty_lineages

    repeated_numeric_episodes = sum(item["repeated_numeric_trajectory"] for item in flagged)
    repeated_casualty_episodes = sum(item["repeated_casualty_trajectory"] for item in flagged)

    report = {
        "numeric_audit_schema_version": NUMERIC_AUDIT_SCHEMA_VERSION,
        "source": _source_metadata(source_path),
        "dataset": {
            "episodes": total,
            "lineages": len({str(item["lineage_id"]) for item in flagged}),
            "future_revised": total_revised,
            "future_revised_rate": total_revised / total,
        },
        "categories": categories,
        "trajectories": {
            "numeric_change_lineages": len(lineage_numeric_counts),
            "repeated_numeric_lineages": len(repeated_numeric_lineages),
            "episodes_in_repeated_numeric_lineages": repeated_numeric_episodes,
            "repeated_numeric_episode_rate": repeated_numeric_episodes / total,
            "casualty_change_lineages": len(lineage_casualty_counts),
            "repeated_casualty_lineages": len(repeated_casualty_lineages),
            "episodes_in_repeated_casualty_lineages": repeated_casualty_episodes,
            "repeated_casualty_episode_rate": repeated_casualty_episodes / total,
        },
        "interpretation": _interpretation(categories, total=total),
    }
    return report, flagged


def classify_numeric_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one episode without changing the source record."""

    candidate_a = str(record["candidate_a"])
    candidate_b = str(record["candidate_b"])
    numbers_a = tuple(_normalise_number(value) for value in _NUMBER_PATTERN.findall(candidate_a))
    numbers_b = tuple(_normalise_number(value) for value in _NUMBER_PATTERN.findall(candidate_b))
    contains_number = bool(numbers_a or numbers_b)
    number_changed = contains_number and Counter(numbers_a) != Counter(numbers_b)

    masked_a = _normalise_text(_NUMBER_PATTERN.sub("<NUM>", candidate_a))
    masked_b = _normalise_text(_NUMBER_PATTERN.sub("<NUM>", candidate_b))
    masked_similarity = difflib.SequenceMatcher(a=masked_a, b=masked_b, autojunk=False).ratio()
    number_only_edit = number_changed and masked_a == masked_b
    number_dominant_edit = number_changed and masked_similarity >= 0.95

    combined = f"{candidate_a} {candidate_b}"
    date_or_update = number_changed and bool(
        _MONTH_PATTERN.search(combined) or _UPDATE_PATTERN.search(combined)
    )
    money_or_percentage = number_changed and bool(
        _MONEY_PATTERN.search(combined) or _PERCENT_PATTERN.search(combined)
    )
    sports_numeric = number_changed and bool(_SPORTS_PATTERN.search(combined))
    casualty_count = number_changed and bool(_CASUALTY_PATTERN.search(combined))

    return {
        "episode_id": str(record["episode_id"]),
        "lineage_id": str(record["lineage_id"]),
        "future_revised": bool(record["future_revised"]),
        "contains_number": contains_number,
        "number_changed": number_changed,
        "number_only_edit": number_only_edit,
        "number_dominant_edit": number_dominant_edit,
        "date_or_update_edit": date_or_update,
        "money_or_percentage_edit": money_or_percentage,
        "sports_numeric_edit": sports_numeric,
        "casualty_count_update": casualty_count,
        "repeated_numeric_trajectory": False,
        "repeated_casualty_trajectory": False,
        "numbers_a": list(numbers_a),
        "numbers_b": list(numbers_b),
        "masked_similarity": masked_similarity,
    }


def render_numeric_shortcut_markdown(report: Mapping[str, Any]) -> str:
    """Render the numeric audit as a blog- and review-friendly report."""

    dataset = report["dataset"]
    lines = [
        "# Numeric Shortcut Audit",
        "",
        "## Dataset",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Episodes | {dataset['episodes']:,} |",
        f"| Article lineages | {dataset['lineages']:,} |",
        f"| Overall future-revision rate | {dataset['future_revised_rate']:.4f} |",
        "",
        "## Numeric categories",
        "",
        "| Category | Episodes | Share | Future-revision rate | Rate without category |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["categories"].items():
        lines.append(
            "| {name} | {episodes:,} | {share:.4f} | {rate:.4f} | {without:.4f} |".format(
                name=name.replace("_", " "),
                episodes=values["episodes"],
                share=values["episode_rate"],
                rate=values["future_revised_rate"],
                without=values["future_revised_rate_without_category"],
            )
        )

    trajectories = report["trajectories"]
    lines.extend(
        [
            "",
            "## Repeated trajectories",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Lineages with a numeric change | {trajectories['numeric_change_lineages']:,} |",
            f"| Lineages with repeated numeric changes | {trajectories['repeated_numeric_lineages']:,} |",
            (
                "| Episodes in repeated numeric lineages | "
                f"{trajectories['episodes_in_repeated_numeric_lineages']:,} |"
            ),
            f"| Lineages with a casualty change | {trajectories['casualty_change_lineages']:,} |",
            (
                "| Lineages with repeated casualty changes | "
                f"{trajectories['repeated_casualty_lineages']:,} |"
            ),
            (
                "| Episodes in repeated casualty lineages | "
                f"{trajectories['episodes_in_repeated_casualty_lineages']:,} |"
            ),
            "",
            "## Interpretation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.extend(
        [
            "",
            "## Required controls",
            "",
            "1. Report the primary result on all episodes.",
            "2. Repeat after excluding casualty-count updates.",
            "3. Repeat after excluding all number-dominant edits.",
            "4. Repeat after replacing every numeric expression with `<NUM>`.",
            "5. Compare against a numeric-features-only future baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def _category_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    name: str,
    total: int,
    total_revised: int,
) -> dict[str, Any]:
    selected = [record for record in records if bool(record[name])]
    revised = sum(bool(record["future_revised"]) for record in selected)
    excluded_total = total - len(selected)
    excluded_revised = total_revised - revised
    return {
        "episodes": len(selected),
        "episode_rate": len(selected) / total,
        "lineages": len({str(record["lineage_id"]) for record in selected}),
        "future_revised": revised,
        "future_revised_rate": revised / len(selected) if selected else 0.0,
        "future_revised_rate_without_category": (
            excluded_revised / excluded_total if excluded_total else 0.0
        ),
        "risk_ratio_vs_complement": _risk_ratio(
            revised,
            len(selected),
            excluded_revised,
            excluded_total,
        ),
        "future_revised_wilson_95": _wilson_interval(revised, len(selected)),
    }


def _interpretation(categories: Mapping[str, Mapping[str, Any]], *, total: int) -> list[str]:
    casualty = categories["casualty_count_update"]
    changed = categories["number_changed"]
    dominant = categories["number_dominant_edit"]
    statements = []
    if casualty["episode_rate"] < 0.02:
        statements.append(
            "Repeated casualty-style updates are present but too rare to dominate the complete dataset."
        )
    if changed["risk_ratio_vs_complement"] > 1.25:
        statements.append(
            "Changed numerical claims are substantially more likely than non-numeric edits to change again."
        )
    if dominant["episodes"]:
        statements.append(
            "Number-dominant edits form a measurable shortcut class and require exclusion and masking ablations."
        )
    statements.append(
        "A surviving transfer result must outperform numeric-only and temporal-position baselines."
    )
    statements.append(f"All rates are computed over {total:,} extracted episodes.")
    return statements


def _normalise_number(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", "").casefold()


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _risk_ratio(
    selected_revised: int,
    selected_total: int,
    complement_revised: int,
    complement_total: int,
) -> float | None:
    if not selected_total or not complement_total or not complement_revised:
        return None
    return (selected_revised / selected_total) / (complement_revised / complement_total)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    if total == 0:
        return None
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"lower": center - half_width, "upper": center + half_width}


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
