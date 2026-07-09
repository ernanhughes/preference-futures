# Step 1 — Freeze Article-Grouped Split Manifests

## Question

Can the later representation experiment be evaluated without allowing versions of the same article to appear in both training and test data?

This is the first confirmatory step because every later score is invalid if the same evolving article crosses partition boundaries.

## Why row-level splitting would fail

A NewsEdits article may contribute several V0→V1→V2 episodes. A random episode split could place an early revision from one article in training and a later revision from the same article in test.

The model could then exploit:

- repeated names and entities;
- recurring article-specific wording;
- the same factual event at different stages;
- exact or reversed candidate pairs;
- later versions of a trajectory it has partly seen already.

The split unit must therefore be:

```text
lineage_id
```

not the individual episode.

## Frozen policy

The primary experiment uses ten deterministic outer buckets.

For outer fold `i`:

```text
test       = bucket i
validation = bucket (i + 1) mod 10
train      = all other buckets
```

This creates an 80/10/10 train/validation/test structure.

Across the ten folds:

- every article lineage is test exactly once;
- every article lineage is validation exactly once;
- no lineage appears in more than one partition within a fold;
- every episode follows its article lineage;
- numeric shortcut prevalence is balanced when numeric flags are supplied.

## Reproduction command

Run this after the context and numeric audits:

```powershell
.\scripts\40-build-grouped-splits.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -NumericFlagsPath artifacts\newsedits\viability-5000\numeric-flags.jsonl `
  -OutputDirectory artifacts\transfer\splits `
  -Folds 10 `
  -Seed 17
```

## Output artifacts

```text
artifacts/transfer/splits/
├── manifest.json
├── split-summary.json
├── split-summary.md
├── fold-00.json
├── fold-01.json
├── fold-02.json
├── fold-03.json
├── fold-04.json
├── fold-05.json
├── fold-06.json
├── fold-07.json
├── fold-08.json
└── fold-09.json
```

The fold files contain lineage IDs and partition summaries only. They do not duplicate sentence text.

The manifest records:

- episode artifact path, size and SHA-256;
- numeric-flag artifact path, size and SHA-256;
- random seed;
- fold policy;
- every lineage’s outer-fold assignment;
- train, validation and test counts for every fold;
- future-label balance;
- candidate-orientation balance;
- numerical-change, number-dominant and casualty-count balance;
- leakage and coverage gates.

## Pass criteria

The step passes only when all generated gates are true:

```text
all lineages tested exactly once
all lineages validated exactly once
test episode share within two percentage points of 10%
test lineage share within two percentage points of 10%
test future-revision rate within three percentage points of the global rate
test numerical-change rate within three percentage points of the global rate
```

The code also asserts that train, validation and test lineage sets are disjoint and complete inside every fold.

## What this step proves

A passing result supports this limited claim:

> The later model comparison can be performed on deterministic article-grouped partitions without direct article-lineage leakage, while preserving approximately comparable target and shortcut distributions across outer test folds.

It does not prove preference-to-future transfer.

It establishes the evaluation boundary within which that claim can be tested.

## What would fail or revise the step

The policy must be revised before model training if:

- any lineage appears in more than one partition within a fold;
- any lineage is never tested or is tested more than once;
- target balance differs sharply across test folds;
- numerical shortcut prevalence is concentrated in a small number of folds;
- a few large article lineages dominate test-fold episode counts;
- changing only the split seed materially changes later conclusions.

The primary seed remains `17`. Other seeds are sensitivity analyses, not opportunities to select a favourable result.

## Results

Run the command, then copy the generated table and gate results from:

```text
artifacts/transfer/splits/split-summary.md
```

into this section.

### Result status

```text
PENDING LOCAL RUN
```

### Generated headline values

```text
Total episodes:                 PENDING
Total article lineages:         PENDING
Outer folds:                    10
Maximum test episode deviation: PENDING
Maximum test target deviation:  PENDING
Maximum numeric-rate deviation: PENDING
All leakage gates:              PENDING
```

## Next step

After the split manifests pass, Step 2 builds the exact training corpora for:

- authentic preference prediction;
- compute-matched language adaptation;
- pair exposure;
- generic temporal-direction prediction;
- random labels;
- shuffled authentic-looking preferences.

No model training should begin before the Step 1 manifest and hashes are frozen.
