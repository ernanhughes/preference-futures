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

The adapter is dependency-free and performs:

- SQLite schema discovery;
- read-only article-version loading;
- deterministic article sampling;
- conservative one-to-one V0→V1 sentence matching;
- V1→V2 revised/stable outcome resolution;
- explicit exclusion counting.

## PowerShell workflow

The numbered scripts in `scripts/` are the normal way to run the repository on Windows.

First-time setup:

```powershell
.\scripts\00-setup.ps1
```

Run tests and linting:

```powershell
.\scripts\01-check.ps1
```

Run the current end-to-end smoke workflow:

```powershell
.\scripts\20-current-smoke-pipeline.ps1 `
  -DatabasePath C:\data\newsedits.db
```

Run the full extraction after the smoke output has been reviewed:

```powershell
.\scripts\12-newsedits-full.ps1 `
  -DatabasePath C:\data\newsedits.db
```

See [`scripts/README.md`](scripts/README.md) for every script and parameter.

## Repository sequence

```text
1. Canonical episode contract       done
2. NewsEdits source adapter         implemented
3. PowerShell run scripts           implemented
4. Context viability audit          next
5. Grouped split manifests
6. Preference-task baselines
7. Frozen representation transfer
8. Sample-efficiency and shortcut controls
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
