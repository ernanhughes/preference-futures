# PowerShell workflows

Run these scripts from the repository root in PowerShell. Every script resolves paths relative to the repository unless an absolute path is supplied. Orchestration scripts stop immediately when a child command or script fails.

## Script index

| Script | Purpose |
|---|---|
| `00-setup.ps1` | Create the virtual environment and install development dependencies. |
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
| `_common.ps1` | Shared internal helpers; do not run directly. |

## Setup and repository checks

```powershell
.\scripts\00-setup.ps1
.\scripts\01-check.ps1
```

Recreate the environment when necessary:

```powershell
.\scripts\00-setup.ps1 -Recreate
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

## Step 1: freeze grouped evaluation splits

Generate the numerical flags first, then build the grouped manifest:

```powershell
.\scripts\15-numeric-shortcut-audit.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -OutputDirectory artifacts\newsedits\viability-5000

.\scripts\40-build-grouped-splits.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -NumericFlagsPath artifacts\newsedits\viability-5000\numeric-flags.jsonl `
  -OutputDirectory artifacts\transfer\splits `
  -Folds 10 `
  -Seed 17
```

The builder invokes the independent verifier automatically. It writes:

```text
artifacts/transfer/splits/manifest.json
artifacts/transfer/splits/split-summary.json
artifacts/transfer/splits/split-summary.md
artifacts/transfer/splits/split-verification.json
artifacts/transfer/splits/split-verification.md
artifacts/transfer/splits/fold-00.json
...
artifacts/transfer/splits/fold-09.json
```

Verify an existing manifest without changing assignments:

```powershell
.\scripts\41-verify-grouped-splits.ps1 `
  -ManifestPath artifacts\transfer\splits\manifest.json
```

For outer fold `i`, test is bucket `i`, validation is bucket `(i + 1) mod 10`, and the remaining eight buckets train. Downstream stages must consume these assignments rather than create a new row-level split.

## Step 2: build compute-matched source corpora

Run Step 2 only after the Step 1 manifest has passed verification:

```powershell
.\scripts\50-build-compute-matched-corpora.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db" `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -SplitsDirectory artifacts\transfer\splits `
  -OutputDirectory artifacts\transfer\corpora `
  -SourceName nyt `
  -Seed 17 `
  -TemporalMaxArticles 20000 `
  -TemporalPoolMultiplier 2.0
```

The command materialises six trained comparison regimes for every fold:

```text
language_adaptation
pair_exposure
temporal_direction
random_label
shuffled_preference
authentic_preference
```

The untouched pretrained encoder is the seventh arm and therefore has no Step 2 corpus.

The builder writes compact source-task instructions rather than duplicating the sentence text:

```text
artifacts/transfer/corpora/
├── manifest.json
├── corpus-summary.md
├── corpus-verification.json
├── corpus-verification.md
├── temporal-pairs.jsonl
├── temporal-pairs-audit.json
├── fold-00/
│   ├── language_adaptation/{train,validation}.jsonl
│   ├── pair_exposure/{train,validation}.jsonl
│   ├── temporal_direction/{train,validation}.jsonl
│   ├── random_label/{train,validation}.jsonl
│   ├── shuffled_preference/{train,validation}.jsonl
│   └── authentic_preference/{train,validation}.jsonl
├── ...
└── fold-09/
```

Ten folds, two source-task partitions and six trained regimes produce exactly:

```text
10 × 2 × 6 = 120 corpus JSONL files
```

Verify an existing corpus directory without rebuilding it:

```powershell
.\scripts\51-verify-compute-matched-corpora.ps1 `
  -OutputDirectory artifacts\transfer\corpora
```

The Step 2 gates require:

- equal record counts across all six regimes inside every fold partition;
- no preference-derived source record from the fold's test lineages;
- no future or V2 field in any source-task record;
- balanced random, pair-exposure and temporal candidate labels;
- different-lineage donors for negative pairs and shuffled preference labels;
- a temporal pool drawn from article lineages disjoint from future evaluation;
- source hashes and all persisted line counts to survive independent verification.

### Temporal-control identification note

On a canonical V0→V1 episode, the retained sentence is also the newer sentence. An exact-pair temporal target would therefore duplicate the authentic target rather than provide an independent control.

Step 2 resolves this by extracting temporal-direction examples from other NewsEdits article lineages that never enter the preference-future evaluation set. Increase `-TemporalMaxArticles` when the external pool cannot meet the frozen train and validation budgets; do not lower the target merely to make the command pass.

### Record matching versus compute matching

Step 2 matches source-record counts and records approximate text exposure. Step 3 must enforce actual compute equality using one fixed encoder checkpoint, tokenizer, maximum sequence length, padding policy, batch size, optimiser, learning-rate schedule, update count, precision and checkpoint rule.

The detailed protocols are:

- [`docs/experiments/01-grouped-split-manifests.md`](../docs/experiments/01-grouped-split-manifests.md)
- [`docs/experiments/02-compute-matched-corpora.md`](../docs/experiments/02-compute-matched-corpora.md)

The matching publication blocks are under [`docs/blog/blocks/`](../docs/blog/blocks/).
