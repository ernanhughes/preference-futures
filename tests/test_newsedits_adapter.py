from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from preference_futures.newsedits import (
    ExtractionConfig,
    discover_article_schema,
    extract_from_database,
    sentence_future_map,
)
from preference_futures.newsedits.cli import main


def _build_fixture_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE article_versions (
                SOURCE TEXT NOT NULL,
                A_ID TEXT NOT NULL,
                VERSION_ID TEXT NOT NULL,
                TEXT TEXT NOT NULL,
                CREATED TEXT,
                TITLE TEXT
            )
            """
        )
        rows = [
            (
                "nyt",
                "article-revised",
                "1",
                "The opening sentence remains unchanged. "
                "The committee may meet on Monday. "
                "The closing sentence remains unchanged.",
                "2026-01-01T10:00:00",
                "Revised example",
            ),
            (
                "nyt",
                "article-revised",
                "2",
                "The opening sentence remains unchanged. "
                "The committee will meet on Monday. "
                "The closing sentence remains unchanged.",
                "2026-01-01T10:05:00",
                "Revised example",
            ),
            (
                "nyt",
                "article-revised",
                "3",
                "The opening sentence remains unchanged. "
                "The committee will meet on Tuesday. "
                "The closing sentence remains unchanged.",
                "2026-01-01T10:10:00",
                "Revised example",
            ),
            (
                "nyt",
                "article-stable",
                "1",
                "The first paragraph remains in place. "
                "The vote could happen later today. "
                "The final paragraph remains in place.",
                "2026-01-02T10:00:00",
                "Stable example",
            ),
            (
                "nyt",
                "article-stable",
                "2",
                "The first paragraph remains in place. "
                "The vote will happen later today. "
                "The final paragraph remains in place.",
                "2026-01-02T10:05:00",
                "Stable example",
            ),
            (
                "nyt",
                "article-stable",
                "3",
                "The first paragraph remains in place. "
                "The vote will happen later today. "
                "The final paragraph remains in place.",
                "2026-01-02T10:10:00",
                "Stable example",
            ),
            (
                "nyt",
                "article-ambiguous",
                "1",
                "The introduction remains unchanged. "
                "Officials released a detailed statement Monday. "
                "The ending remains unchanged.",
                "2026-01-03T10:00:00",
                "Ambiguous example",
            ),
            (
                "nyt",
                "article-ambiguous",
                "2",
                "The introduction remains unchanged. "
                "Officials released a statement Monday. "
                "It included several additional details. "
                "The ending remains unchanged.",
                "2026-01-03T10:05:00",
                "Ambiguous example",
            ),
            (
                "nyt",
                "article-ambiguous",
                "3",
                "The introduction remains unchanged. "
                "Officials released a statement Monday. "
                "It included several additional details. "
                "The ending remains unchanged.",
                "2026-01-03T10:10:00",
                "Ambiguous example",
            ),
        ]
        connection.executemany(
            "INSERT INTO article_versions VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def test_schema_discovery_and_fixture_extraction(tmp_path: Path) -> None:
    database_path = tmp_path / "newsedits.sqlite"
    _build_fixture_database(database_path)

    connection = sqlite3.connect(database_path)
    try:
        schema = discover_article_schema(connection)
        result = extract_from_database(
            connection,
            schema,
            config=ExtractionConfig(min_sentence_chars=15),
            seed=7,
        )
    finally:
        connection.close()

    assert schema.table == "article_versions"
    assert schema.article_id == "A_ID"
    assert len(result.examples) == 2
    outcomes = {
        example.article_id: example.build_episode(seed=7).future_revised
        for example in result.examples
    }
    assert outcomes == {"article-revised": True, "article-stable": False}
    assert result.audit.articles_seen == 3
    assert result.audit.accepted_examples == 2
    assert result.audit.exclusions["ambiguous_replacement"] == 1


def test_sentence_future_map_marks_deleted_or_ambiguous_branch_as_changed() -> None:
    mapped = sentence_future_map(
        ["Opening remains.", "The selected sentence remains here.", "Closing remains."],
        ["Opening remains.", "Closing remains."],
    )

    assert mapped[1] is None


def test_cli_writes_jsonl_and_audit(tmp_path: Path) -> None:
    database_path = tmp_path / "newsedits.sqlite"
    output_path = tmp_path / "episodes.jsonl"
    audit_path = tmp_path / "audit.json"
    _build_fixture_database(database_path)

    exit_code = main(
        [
            "extract",
            "--db",
            str(database_path),
            "--out",
            str(output_path),
            "--audit-out",
            str(audit_path),
            "--seed",
            "11",
            "--min-sentence-chars",
            "15",
        ]
    )

    assert exit_code == 0
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {record["future_revised"] for record in records} == {True, False}
    assert all(record["newsedits_schema_version"] == 1 for record in records)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["accepted_examples"] == 2
    assert audit["exclusions"]["ambiguous_replacement"] == 1
