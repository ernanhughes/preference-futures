from __future__ import annotations

import sqlite3
from pathlib import Path

from preference_futures.newsedits import (
    ExtractionConfig,
    discover_split_sentence_schema,
    extract_from_split_database,
)


def _build_abbreviation_fixture(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE split_sentences (
                entry_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                sent_idx INTEGER NOT NULL,
                sentence TEXT NOT NULL
            )
            """
        )
        versions = {
            0: [
                "The administration may appoint Mr. Dobbins tomorrow.",
                "The second sentence remains unchanged.",
            ],
            1: [
                "The administration will appoint Mr. Dobbins tomorrow.",
                "The second sentence remains unchanged.",
            ],
            2: [
                "The administration will appoint Mr. Dobbins tomorrow.",
                "The second sentence remains unchanged.",
            ],
        }
        rows = [
            ("entry-abbreviation", version, sentence_index, sentence)
            for version, sentences in versions.items()
            for sentence_index, sentence in enumerate(sentences)
        ]
        connection.executemany("INSERT INTO split_sentences VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()


def test_official_sentence_rows_are_not_resplit_at_honorifics(tmp_path: Path) -> None:
    database_path = tmp_path / "nyt-matched-sentences.db"
    _build_abbreviation_fixture(database_path)

    connection = sqlite3.connect(database_path)
    try:
        schema = discover_split_sentence_schema(connection)
        result = extract_from_split_database(
            connection,
            schema,
            source_name="nyt",
            config=ExtractionConfig(min_sentence_chars=10),
            seed=17,
        )
    finally:
        connection.close()

    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.triplet.v0_sentence == "The administration may appoint Mr. Dobbins tomorrow."
    assert example.triplet.v1_sentence == "The administration will appoint Mr. Dobbins tomorrow."
    assert example.context_after == "The second sentence remains unchanged."
    assert example.future_revised is False

    audit = result.audit.to_record()
    assert audit["articles_seen"] == 1
    assert audit["articles_with_examples"] == 1
    assert audit["accepted_examples"] == 1
    assert audit["future_revised_examples"] == 0
    assert audit["future_stable_examples"] == 1
    assert audit["future_revised_rate"] == 0.0
    assert audit["acceptance_rate"] == 1.0
