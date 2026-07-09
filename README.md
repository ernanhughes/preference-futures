# preference-futures

Research code for testing whether preference-specific learning transfers to prediction of a later, decision-linked outcome.

## Publication and evidence map

- [Executable blog draft](docs/blog/what-does-a-preference-know-about-the-future.md)
- [Research claim ledger](docs/CLAIMS.md)
- [Representation experiment steps](docs/experiments/README.md)
- [PowerShell workflow index](scripts/README.md)

The publication rule is simple: every sentence beginning with “we found” must map to a committed script and output artifact. Representation-transfer claims remain hypotheses until the grouped control experiment is implemented and run.

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

## Reproduce the blog evidence

First-time setup:

```powershell
.\scripts\00-setup.ps1
```

Run the complete current evidence chain against an official source database:

```powershell
.\scripts\30-reproduce-blog-evidence.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

The command performs:

```text
repository checks
→ extraction
→ artifact verification
→ context viability audit
→ numeric shortcut audit
```

It writes all artifacts under:

```text
artifacts/newsedits/blog-evidence/
```

Individual stages remain runnable through the numbered scripts in [`scripts/`](scripts/README.md).

## Current viability checkpoint

The first 5,000-article deterministic NYT extraction produced:

```text
12,056 accepted preference episodes
3,386 article lineages
3,104 revised futures
8,952 stable futures
25.75% future-revision rate
29.88% replacement-opcode acceptance rate
```

The context audit freezes descriptive checks for target balance, candidate orientation, lineage concentration, context availability, source-boundary artifacts, exact candidate-pair reversals, similarity bands, sentence-position bands and article-version bands.

The numeric audit identifies changed values, number-only edits, number-dominant edits, date/update changes, money and percentages, sports values, casualty-count changes and repeated numeric trajectories. These flags become mandatory controls in the later transfer experiment.

## Step 1: grouped evaluation boundary verified

Build and independently verify deterministic article-lineage grouped folds:

```powershell
.\scripts\40-build-grouped-splits.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -NumericFlagsPath artifacts\newsedits\viability-5000\numeric-flags.jsonl `
  -OutputDirectory artifacts\transfer\splits `
  -Folds 10 `
  -Seed 17
```

The seed-17 result assigns all 12,056 episodes from 3,386 article lineages into ten 80/10/10 outer-fold experiments. Every lineage is test exactly once and validation exactly once. Test folds contain 1,204–1,207 episodes from 338–339 lineages.

The maximum test-fold deviations are:

```text
future-revision rate: 0.0840 percentage points
numeric-change rate:  0.0697 percentage points
```

The assignments, source hashes and compact result are frozen in [Step 1](docs/experiments/01-grouped-split-manifests.md) and [`docs/results/step-01-grouped-splits.json`](docs/results/step-01-grouped-splits.json).

## Repository sequence

```text
1. Canonical episode contract       done
2. NewsEdits source adapter         done
3. Official split_sentences input   done
4. PowerShell run scripts           done
5. Context viability audit          verified
6. Numeric shortcut audit           verified
7. Executable blog and claim ledger implemented
8. Grouped split manifests          verified and frozen
9. Compute-matched training corpora next
10. Preference and control training
11. Frozen representation transfer
12. Sample-efficiency controls
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
