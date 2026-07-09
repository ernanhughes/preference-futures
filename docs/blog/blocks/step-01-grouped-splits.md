## Step One: Freeze the Evaluation Boundary

Before training any encoder, we froze the article-grouped train, validation and test assignments.

This is not administrative bookkeeping. It is part of the experiment.

A single NewsEdits article can produce several revision episodes. A row-level random split could place an early version of an article in training and a later version of the same article in test. The model would then see part of the trajectory it was supposedly predicting.

The split unit is therefore the complete article lineage:

```text
lineage_id
```

The repository assigns every lineage to one of ten deterministic outer buckets. For fold `i`:

```text
test       = bucket i
validation = bucket (i + 1) mod 10
train      = the remaining eight buckets
```

That produces an 80/10/10 structure while ensuring that every article is test exactly once and validation exactly once across the ten confirmatory folds.

Run:

```powershell
.\scripts\40-build-grouped-splits.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -NumericFlagsPath artifacts\newsedits\viability-5000\numeric-flags.jsonl `
  -OutputDirectory artifacts\transfer\splits `
  -Folds 10 `
  -Seed 17
```

The command writes one manifest and ten fold files. The files contain lineage assignments and balance statistics, not duplicated article text.

The split is accepted only if:

- train, validation and test lineages are disjoint in every fold;
- every lineage is tested exactly once;
- every lineage is used for validation exactly once;
- test-fold episode shares remain close to 10%;
- future-revision rates remain close to the full dataset rate;
- numerical-change prevalence remains balanced across test folds.

### Result

<!-- Replace this block with the generated values from artifacts/transfer/splits/split-summary.md. -->

```text
Status:                        PENDING LOCAL RUN
Episodes:                      PENDING
Article lineages:              PENDING
Outer folds:                   10
Maximum target-rate deviation: PENDING
Maximum numeric-rate deviation:PENDING
Leakage and coverage gates:    PENDING
```

A passing result proves only this:

> The later model comparison uses deterministic article-grouped partitions without direct lineage leakage and without concentrating the primary target or the known numerical shortcut in one test fold.

It does not prove that preference learning transfers to future prediction.

It makes that claim testable.
