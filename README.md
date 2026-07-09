# preference-futures

Research code for testing whether preference-specific learning transfers to prediction of a later, decision-linked outcome.

## Episode contract

A revision triplet is represented as:

```text
V0: rejected sentence
V1: selected replacement
V2: later state of the selected replacement, or no unambiguous continuation
```

The episode builder randomises whether V1 appears as candidate A or B, while preserving both the preference label and the future outcome attached to V1. A missing one-to-one V2 continuation counts as revised or removed.

## NewsEdits adapter

The old experiment scripts remain in `src/preference_futures/newsedits/` as research provenance. New ingestion code uses the importable package API:

```python
from preference_futures.newsedits import (
    discover_article_schema,
    extract_from_database,
)
```

Inspect a NewsEdits SQLite database:

```bash
python -m preference_futures.newsedits inspect --db /path/to/newsedits.db
```

Extract versioned JSONL episodes and an exclusion audit:

```bash
python -m preference_futures.newsedits extract \
  --db /path/to/newsedits.db \
  --out artifacts/newsedits/episodes.jsonl \
  --audit-out artifacts/newsedits/audit.json \
  --seed 17
```

The adapter is dependency-free and performs:

- SQLite schema discovery;
- read-only article-version loading;
- deterministic article sampling;
- conservative one-to-one V0→V1 sentence matching;
- V1→V2 revised/stable outcome resolution;
- explicit exclusion counting.

## Repository sequence

```text
1. Canonical episode contract       done
2. NewsEdits source adapter         implemented
3. Context viability audit          next
4. Grouped split manifests
5. Preference-task baselines
6. Frozen representation transfer
7. Sample-efficiency and shortcut controls
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
