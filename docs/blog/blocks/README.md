# Executable Blog Blocks

These sections extend the main draft as each representation experiment becomes executable and then verified.

## Order

| Step | Block | Status |
|---:|---|---|
| 1 | [`step-01-grouped-splits.md`](step-01-grouped-splits.md) | Verified |
| 2 | [`step-02-compute-matched-corpora.md`](step-02-compute-matched-corpora.md) | Verified |
| 3 | [`step-03-fixed-budget-training.md`](step-03-fixed-budget-training.md) | Implemented; awaiting local smoke run |

## Publication rule

A block moves from procedure to finding only after its committed command has been run on the frozen artifacts and the generated result has been recorded in both:

```text
docs/experiments/
docs/results/
```

Step 2 records 24,112 independent temporal pairs, 5,135 external lineages, 120 persisted corpus files and 651,024 verified source-task records, with every builder and persisted-verification gate passing.

Step 3 now defines the single model snapshot, fixed padded-token budget, fixed update count and final-checkpoint rule used to train all six regimes. Its result remains procedural until the model snapshot and six-regime smoke artifacts exist.

## Main-draft insertion points

The Step 2 block replaces the weaker control outline under:

```text
## What Authentic Preference Learning Must Beat
```

It supersedes the idea that temporal direction can be trained independently on the exact V0→V1 preference pairs. On those pairs, retained identity and chronological newness are the same label by construction.

The Step 3 block follows Step 2 and turns those frozen corpora into trained representations without yet making a future-transfer claim.
