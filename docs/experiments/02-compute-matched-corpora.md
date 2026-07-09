# Step 2 — Build the Compute-Matched Source Corpora

## Question

Can we construct the authentic preference source task and its strongest alternatives before training, using the frozen Step 1 lineages, without leaking the future target and without giving one trained regime more source examples than another?

Step 2 is a data-contract step. It does not train an encoder and it does not test future transfer.

It freezes exactly what each source-task encoder will be trained to do.

## The temporal-control correction

The first design proposed training temporal direction on the same V0→V1 pairs used for authentic preference prediction.

That is not an independent control.

For these revision-derived episodes:

```text
V0 = earlier rejected sentence
V1 = later retained sentence
```

After candidate-order randomisation, both source tasks have the same target:

```text
which candidate did the editor retain?
=
which candidate is newer?
```

An exact-pair temporal-direction model would therefore receive the same inputs and labels as the authentic preference model. Renaming the target would not create a different experiment.

The independent temporal-direction corpus is instead extracted from other NewsEdits article lineages that never appear in the preference-future evaluation set. It uses the same publication domain and revision mechanism, but it cannot leak an evaluation article trajectory.

This does not fully identify a uniquely preference-specific mechanism. A later positive result must still be described as transfer from authentic revision-choice supervision unless it beats the independent temporal representation and generalises to datasets where preference is not synonymous with chronological replacement.

## Frozen inputs

```text
artifacts/newsedits/viability-5000/episodes.jsonl
artifacts/transfer/splits/manifest.json
artifacts/transfer/splits/fold-00.json
...
artifacts/transfer/splits/fold-09.json
E:/data/newsedits/nyt-matched-sentences.db
```

The episode and split hashes from Step 1 remain authoritative. Step 2 does not create a new row-level or lineage-level split.

## The seven comparison arms

Six encoders receive additional source-task training. The untouched generic encoder is retained as a seventh comparison arm.

### Generic encoder

No additional training. This arm has no Step 2 corpus.

### Compute-matched language adaptation

The encoder receives the same canonical NewsEdits preference episodes, but the objective is deterministic masked-word reconstruction rather than preference prediction.

This controls for additional NewsEdits language and domain adaptation.

### Pair-exposure representation

The encoder predicts whether the two candidates originate from the same revision episode.

Half the records retain the true candidate pair. Half replace candidate B with candidate B from a deterministic different-lineage donor episode. The donor mapping is a permutation, preserving complete candidate exposure while removing preference supervision.

### Independent temporal-direction representation

The encoder predicts which candidate is newer using one-to-one replacements extracted from NewsEdits articles outside all evaluation lineages.

The temporal pool is:

- source-matched;
- article-lineage disjoint from future evaluation;
- grouped into deterministic temporal outer buckets;
- record-count matched to each authentic fold partition;
- approximately matched to the authentic input-length distribution.

### Random-label representation

The encoder sees the authentic pair-and-context inputs but receives deterministic balanced labels unrelated to the editor’s choice.

### Shuffled-preference representation

The encoder sees each authentic pair-and-context input, but its target is donated by a different episode from a different article lineage.

The authentic label multiset is preserved exactly within every train and validation partition while the link between choice and article state is broken.

### Authentic preference representation

The encoder predicts which candidate the editor retained. It sees no V2 sentence, future-revision label or future outcome field.

## Reproduction command

```powershell
.\scripts\50-build-compute-matched-corpora.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db" `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -SplitsDirectory artifacts\transfer\splits `
  -OutputDirectory artifacts\transfer\corpora `
  -Seed 17 `
  -TemporalMaxArticles 20000 `
  -TemporalPoolMultiplier 2.0
```

The script:

```text
validates the frozen Step 1 assignments
→ extracts an external temporal-direction pool
→ builds six train corpora for every outer fold
→ builds six source-validation corpora for every outer fold
→ writes a corpus manifest and summary
→ reopens every artifact and verifies it independently
```

## Output artifacts

```text
artifacts/transfer/corpora/
├── manifest.json
├── corpus-summary.md
├── corpus-verification.json
├── corpus-verification.md
├── temporal-pairs.jsonl
├── temporal-pairs-audit.json
├── fold-00/
│   ├── authentic_preference/{train,validation}.jsonl
│   ├── language_adaptation/{train,validation}.jsonl
│   ├── pair_exposure/{train,validation}.jsonl
│   ├── temporal_direction/{train,validation}.jsonl
│   ├── random_label/{train,validation}.jsonl
│   └── shuffled_preference/{train,validation}.jsonl
├── ...
└── fold-09/
```

For ten folds, two source-task partitions and six trained regimes:

```text
10 × 2 × 6 = 120 corpus JSONL files
```

## Forbidden leakage

No Step 2 source-task JSONL record may contain:

```text
future_revised
future_stable
future_label
future_outcome
v2_sentence
v2_version_id
```

The future target remains unavailable until the frozen source encoders are evaluated in the later probe stage.

## Frozen pass criteria

The builder must establish:

```text
all six trained corpora have equal record counts within every fold partition
no preference-derived source record uses the fold’s test lineages
no source-task record contains future or V2 fields
random labels are balanced
pair-exposure labels are balanced
negative pair donors come from different lineages
shuffled preference preserves authentic label counts
shuffled-label donors come from different lineages
temporal pairs are external to all evaluation lineages
temporal-direction candidate orientation is balanced
```

The persisted verifier separately checks:

```text
all 120 expected corpus files exist
persisted line counts agree with the manifest
record corpus, fold and partition fields agree with their paths
source IDs are unique inside each corpus file
source hashes still match
future fields remain absent after writing
external temporal pool and audit artifacts exist
```

## What “compute matched” means here

Step 2 matches the number of source-task records and records approximate text exposure.

That is necessary but not sufficient for equal compute.

Step 3 must enforce:

```text
same starting encoder checkpoint
same tokenizer
same frozen maximum sequence length
same fixed padding policy
same batch size
same optimiser
same learning-rate schedule
same number of update steps
same precision
same gradient-accumulation rule
same fixed checkpoint schedule
no task-specific early stopping
```

Different objectives produce different losses, but they must not receive different opportunities to update the encoder.

## Verified result

The seed-17 real-data run passed every builder and persisted-artifact gate.

### Dataset and external temporal pool

| Measure | Result |
|---|---:|
| Preference episodes | 12,056 |
| Evaluation article lineages | 3,386 |
| Independent temporal pairs | 24,112 |
| Independent temporal lineages | 5,135 |
| Evaluation-lineage overlap | 0 |
| External articles selected | 20,000 |
| External articles read before target reached | 6,849 |
| Replacement opcodes examined | 44,840 |

The temporal extractor reached exactly twice the preference-episode count, and all three temporal-pool gates passed:

```text
target pair count reached
temporal lineages disjoint from evaluation
temporal pair IDs unique
```

### Fold budgets

Each trained regime receives the same record count inside a given fold partition.

```text
train records per corpus:      9,643–9,646
validation records per corpus: 1,204–1,207
```

Across all folds, partitions and trained regimes, the verifier read:

```text
651,024 persisted source-task records
```

### Persisted verification

```text
120 expected corpus files
120 observed corpus files
all source hashes matched
all record counts matched
all record identities matched their paths
all source IDs were unique within each file
no future or V2 field was present
no verification errors
```

### Exposure audit

The authentic, language-adaptation, random-label and shuffled-preference regimes use the same episode text and therefore have identical whitespace-token exposure within each partition.

Pair-exposure differs only through its cross-lineage candidate-B substitutions; its mean exposure difference from authentic preference is negligible.

The external temporal-direction corpus is longer:

```text
train exposure above authentic:
  minimum 6.5112%
  maximum 7.4736%
  mean    7.0508%

validation exposure above authentic:
  minimum 3.7020%
  maximum 10.7591%
  mean    7.0872%
```

This is not a Step 2 failure because record budgets, lineage separation and labels are correct. It is a Step 3 constraint: fixed maximum sequence length, padding, batch size and update count must prevent the temporal arm from receiving more optimisation compute merely because its sentences are longer.

### Frozen source identities

```text
episodes SHA-256:
df4e40330ad6d3f6d4977e1630e2e54e3cfc06b01277d1aa98b7994e8c63e5ab

split manifest SHA-256:
77864f4e0efae5fd98e75998b15b03cab026913025fca6d708c3b433e7886faf

temporal pairs SHA-256:
6a93a3a2cb0d41f1f1e0941e3406e817fe2418ac870e02e3eb648d149e25ee92

NewsEdits database SHA-256:
1b81497d415b9dd86134f0871e73b6dde096bcd4d084f0f54ae942cb3db86ace
```

The compact machine-readable record is [`docs/results/step-02-compute-matched-corpora.json`](../results/step-02-compute-matched-corpora.json).

## Supported claim

The verified result supports this limited statement:

> The authentic revision-choice objective and five trained alternatives were frozen before model training with equal source-record budgets, no direct future-label leakage and an independent temporal-control pool.

It does not prove equal optimisation compute before Step 3 enforces the trainer contract.

It does not prove that authentic preference training improves future prediction.

It does not prove that any later transfer is uniquely preference-specific rather than accepted-revision-specific.

## Next step

Step 3 implements one trainer that consumes these frozen manifests and trains all six additional encoders under the same optimisation budget.

The trainer must not inspect future labels, future-probe scores or test-fold outcomes while selecting source-task checkpoints.
