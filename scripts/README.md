# PowerShell workflows

Run these scripts from the repository root in PowerShell. Every script resolves paths relative to the repository unless an absolute path is supplied.

## Script list

| Script | Purpose |
|---|---|
| `00-setup.ps1` | Create `.venv` and install the project with development dependencies. |
| `01-check.ps1` | Parse all PowerShell files, then run pytest and Ruff. |
| `02-parse-powershell.ps1` | Standalone syntax validation for every PowerShell script. |
| `10-newsedits-inspect.ps1` | Inspect SQLite tables and detect the NewsEdits article schema. |
| `11-newsedits-smoke.ps1` | Extract a bounded sample and immediately verify its artifacts. |
| `12-newsedits-full.ps1` | Extract the requested full or bounded production dataset. |
| `13-newsedits-verify.ps1` | Stream-validate JSONL episodes against the extraction audit. |
| `14-context-viability-audit.ps1` | Audit class balance, contexts, lineage concentration, artifacts and pair reversals. |
| `20-current-smoke-pipeline.ps1` | Run checks, inspect the database, extract a smoke sample, and verify it. |
| `_common.ps1` | Shared internal helpers; do not run directly. |

## First-time setup

```powershell
.\scripts\00-setup.ps1
```

Recreate the virtual environment:

```powershell
.\scripts\00-setup.ps1 -Recreate
```

## Validate the repository

Parse PowerShell only:

```powershell
.\scripts\02-parse-powershell.ps1
```

Run all repository checks:

```powershell
.\scripts\01-check.ps1
```

## Recommended first real run

```powershell
.\scripts\20-current-smoke-pipeline.ps1 `
  -DatabasePath C:\data\newsedits.db
```

This creates:

```text
artifacts/newsedits/smoke/episodes.jsonl
artifacts/newsedits/smoke/audit.json
```

Filter to one or more sources:

```powershell
.\scripts\20-current-smoke-pipeline.ps1 `
  -DatabasePath C:\data\newsedits.db `
  -Sources "nyt,washington_post" `
  -MaxArticles 500 `
  -MaxExamples 5000
```

Specify a non-default article table:

```powershell
.\scripts\10-newsedits-inspect.ps1 `
  -DatabasePath C:\data\newsedits.db `
  -Table article_versions
```

## Full extraction

`MaxArticles = 0` and `MaxExamples = 0` mean no limit.

```powershell
.\scripts\12-newsedits-full.ps1 `
  -DatabasePath C:\data\newsedits.db `
  -OutputDirectory artifacts\newsedits\full `
  -Seed 17
```

Use bounded values while tuning extraction rules:

```powershell
.\scripts\12-newsedits-full.ps1 `
  -DatabasePath C:\data\newsedits.db `
  -MaxArticles 10000 `
  -MaxExamples 100000
```

## Verify existing outputs

```powershell
.\scripts\13-newsedits-verify.ps1 `
  -EpisodesPath artifacts\newsedits\smoke\episodes.jsonl `
  -AuditPath artifacts\newsedits\smoke\audit.json
```

## Audit context viability

Run this after extraction and verification, before creating train/test splits:

```powershell
.\scripts\14-context-viability-audit.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -OutputDirectory artifacts\newsedits\viability-5000
```

This writes:

```text
artifacts/newsedits/viability-5000/context-viability.json
artifacts/newsedits/viability-5000/context-viability.md
```

The audit records target balance, candidate-order balance, lineage concentration, context availability, residual source-boundary artifacts, boilerplate flags, exact candidate-pair reversals, similarity bands, sentence-position bands and V1-version bands. A nonzero exit code means at least one frozen viability gate failed.

All orchestration scripts stop immediately when a child command or script fails. New numbered scripts will be added as split manifests, baselines, and transfer experiments become executable. Scripts should wrap importable package commands rather than contain research logic themselves.
