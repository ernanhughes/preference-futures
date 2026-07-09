# PowerShell workflows

Run these scripts from the repository root in PowerShell. Every script resolves paths relative to the repository unless an absolute path is supplied. Orchestration scripts stop immediately when a child command or script fails.

## Script index

| Script | Purpose |
|---|---|
| `00-setup.ps1` | Create the virtual environment; add `-Training` for PyTorch and Transformers. |
| `01-check.ps1` | Parse all PowerShell files, then run pytest and Ruff. |
| `02-parse-powershell.ps1` | Validate the syntax of every PowerShell script. |
| `10-newsedits-inspect.ps1` | Inspect SQLite tables and detect the NewsEdits schema. |
| `11-newsedits-smoke.ps1` | Extract a bounded sample and verify it. |
| `12-newsedits-full.ps1` | Extract the requested production dataset. |
| `13-newsedits-verify.ps1` | Stream-validate extracted episodes against the audit. |
| `14-context-viability-audit.ps1` | Audit target balance, contexts, lineages, artifacts and pair reversals. |
| `15-numeric-shortcut-audit.ps1` | Measure numerical volatility and write per-episode flags. |
| `20-current-smoke-pipeline.ps1` | Run checks, inspect, extract and verify a smoke sample. |
| `30-reproduce-blog-evidence.ps1` | Reproduce the extraction, context and numerical evidence chain. |
| `40-build-grouped-splits.ps1` | Freeze and verify deterministic article-grouped folds. |
| `41-verify-grouped-splits.ps1` | Independently verify an existing grouped-split manifest. |
| `50-build-compute-matched-corpora.ps1` | Build and verify the six Step 2 source-task corpora. |
| `51-verify-compute-matched-corpora.ps1` | Independently verify persisted Step 2 corpus artifacts. |
| `60-prepare-fixed-budget-training.ps1` | Resolve the immutable model revision and freeze Step 3. |
| `61-step3-smoke.ps1` | Run two non-confirmatory updates for all six regimes on fold 0. |
| `62-train-fixed-budget-representations.ps1` | Run resumable confirmatory Step 3 training. |
| `63-verify-fixed-budget-training.ps1` | Verify persisted confirmatory model runs and budgets. |
| `70-freeze-source-task-encoders.ps1` | Diagnose Step 3 and freeze the seven-arm Step 4 manifest. |
| `80-prepare-frozen-representations.ps1` | Freeze Step 5 input, pooling and matrix contracts. |
| `81-extract-frozen-representations.ps1` | Extract resumable Step 5 matrices. |
| `82-verify-frozen-representations.ps1` | Verify all persisted Step 5 matrices and row identities. |
| `90-prepare-identical-future-probes.ps1` | Freeze Step 6 target, probe and comparison contracts. |
| `91-train-identical-future-probes.ps1` | Train or resume all identical Step 6 probes. |
| `92-verify-identical-future-probes.ps1` | Recompute probes, aggregate out-of-fold forecasts and bootstrap comparisons. |
| `_common.ps1` | Shared internal helpers; do not run directly. |

## Setup and repository checks

```powershell
.\scripts\00-setup.ps1
.\scripts\01-check.ps1
```

Install the optional model stack:

```powershell
.\scripts\00-setup.ps1 -Training
```

Recreate the environment when necessary:

```powershell
.\scripts\00-setup.ps1 -Recreate
.\scripts\00-setup.ps1 -Recreate -Training
```

Run PowerShell parsing alone:

```powershell
.\scripts\02-parse-powershell.ps1
```

## Reproduce the current evidence dataset

```powershell
.\scripts\30-reproduce-blog-evidence.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

The default run creates the deterministic 5,000-article, seed-17 evidence artifacts under:

```text
artifacts/newsedits/blog-evidence/
```

The individual extraction and audit scripts remain available when a stage needs to be inspected separately.
