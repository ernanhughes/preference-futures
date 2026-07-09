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

The build script now runs an independent persisted-manifest verification automatically. An existing manifest can be checked separately with:

```powershell
.\scripts\41-verify-grouped-splits.ps1 `
  -ManifestPath artifacts\transfer\splits\manifest.json
```

## Output artifacts

```text
artifacts/transfer/splits/
├── manifest.json
├── split-summary.json
├── split-summary.md
├── split-verification.json
├── split-verification.md
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

- episode artifact size and SHA-256;
- numeric-flag artifact size and SHA-256;
- random seed;
- fold policy;
- every lineage’s outer-fold assignment;
- train, validation and test counts for every fold;
- future-label balance;
- candidate-orientation balance;
- numerical-change, number-dominant and casualty-count balance;
- leakage and coverage gates.

## Pass criteria

The step passes only when all generated gates and independent checks are true:

```text
all lineages tested exactly once
all lineages validated exactly once
all assignment fold IDs valid
assignment map agrees with test-fold summaries
validation is the next outer test bucket
training is the exact complement of validation and test
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

### Result status

```text
VERIFIED — ALL BUILDER GATES AND INDEPENDENT COVERAGE CHECKS PASSED
```

### Dataset

| Measure | Result |
|---|---:|
| Episodes | 12,056 |
| Article lineages | 3,386 |
| Future revised | 3,104 |
| Future stable | 8,952 |
| Future-revision rate | 25.7465% |
| Selected-B rate | 50.6636% |
| Number-changed rate | 10.1858% |
| Number-dominant rate | 3.5169% |
| Casualty-count rate | 0.8295% |

### Maximum outer test-fold deviations

| Measure | Absolute deviation | Percentage-point equivalent |
|---|---:|---:|
| Episode share from expected 10% | 0.000133 | 0.0133 points |
| Lineage share from expected 10% | 0.000177 | 0.0177 points |
| Future-revision rate from global | 0.000840 | 0.0840 points |
| Numerical-change rate from global | 0.000697 | 0.0697 points |

### Outer test-fold balance

| Fold | Test lineages | Test episodes | Episode share | Revised rate | Number-changed rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 339 | 1,204 | 9.9867% | 25.8306% | 10.2159% |
| 1 | 338 | 1,207 | 10.0116% | 25.6835% | 10.1906% |
| 2 | 338 | 1,206 | 10.0033% | 25.7048% | 10.1161% |
| 3 | 339 | 1,205 | 9.9950% | 25.8091% | 10.2075% |
| 4 | 339 | 1,205 | 9.9950% | 25.7261% | 10.2075% |
| 5 | 339 | 1,206 | 10.0033% | 25.7048% | 10.1990% |
| 6 | 339 | 1,205 | 9.9950% | 25.7261% | 10.2075% |
| 7 | 338 | 1,206 | 10.0033% | 25.7877% | 10.1990% |
| 8 | 339 | 1,206 | 10.0033% | 25.7877% | 10.1161% |
| 9 | 338 | 1,206 | 10.0033% | 25.7048% | 10.1990% |

### Verification

The persisted manifest contains exactly 3,386 unique lineage assignments. Every assignment uses a valid fold ID from 0 through 9. The assignment-map counts agree with each test-fold summary. Across the ten folds:

- test counts sum exactly to all dataset totals;
- validation counts sum exactly to all dataset totals;
- validation for fold `i` is exactly test bucket `(i + 1) mod 10`;
- training counts are exactly total minus validation minus test;
- all six builder gates passed.

The source artifacts frozen by the manifest are:

```text
episodes.jsonl
SHA-256: df4e40330ad6d3f6d4977e1630e2e54e3cfc06b01277d1aa98b7994e8c63e5ab
bytes:   13,149,409

numeric-flags.jsonl
SHA-256: abf517a03760da77bf60029d3385887ec6d3b73bd7db7e3d74f238ead07d75c1
bytes:   5,865,383
```

A compact committed result record is stored at:

```text
docs/results/step-01-grouped-splits.json
```

### Interpretation

The observed deviations are far below the preregistered limits. The largest target-rate difference between an outer test fold and the complete dataset is only **0.084 percentage points**. The largest numerical-change-rate difference is only **0.070 percentage points**.

The folds are therefore not merely acceptable. They are effectively matched on the primary outcome and the strongest known shortcut class while remaining strictly grouped by article lineage.

Step 1 is closed. These assignments and input hashes are frozen for every downstream training, probing, calibration, bootstrap and ablation stage.

## Next step

Step 2 builds the exact training corpora for:

- authentic preference prediction;
- compute-matched language adaptation;
- pair exposure;
- generic temporal-direction prediction;
- random labels;
- shuffled authentic-looking preferences.

No downstream stage may regenerate the split assignments or alter the frozen source artifacts silently.
