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

The adapter accepts both NewsEdits storage forms:

1. a full article-version table containing source, article ID, version ID and article text;
2. official source-specific databases such as `nyt-matched-sentences.db`, reconstructed from the `split_sentences` table.

Official `split_sentences` columns are discovered through aliases for:

```text
entry_id
version
sent_idx
sentence
```

The publisher is inferred from filenames such as `nyt-matched-sentences.db`, or can be supplied explicitly with `--source-name`.

```python
from preference_futures.newsedits import (
    discover_article_schema,
    discover_split_sentence_schema,
    extract_from_database,
    extract_from_split_database,
)
```

The adapter is dependency-free and performs:

- SQLite schema discovery;
- read-only article-version loading or reconstruction;
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

Run the current end-to-end smoke workflow directly against an official database:

```powershell
.\scripts\20-current-smoke-pipeline.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

Run the full extraction after the smoke output has been reviewed:

```powershell
.\scripts\12-newsedits-full.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

See [`scripts/README.md`](scripts/README.md) for every script and parameter.

## Repository sequence

```text
1. Canonical episode contract       done
2. NewsEdits source adapter         implemented
3. Official split_sentences input   implemented
4. PowerShell run scripts           implemented
5. Context viability audit          next
6. Grouped split manifests
7. Preference-task baselines
8. Frozen representation transfer
9. Sample-efficiency and shortcut controls
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
