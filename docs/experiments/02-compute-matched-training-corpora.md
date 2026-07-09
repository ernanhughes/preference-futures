# Step 2 — Build Compute-Matched Training Corpora

## Question

Can the representation experiment give authentic preference learning a fair source-task comparison against controls that see the same examples, the same partitions and the same input text?

This step still does **not** train a model. It freezes the data products that later training scripts must consume.

## Why this step exists

A positive preference-transfer result would be untrustworthy if the authentic preference encoder were the only encoder that received:

- additional NewsEdits domain exposure;
- revision-pair exposure;
- the same article-lineage train/validation/test boundary;
- the same amount of serialized input text;
- the same number of source-task examples.

Therefore Step 2 creates all source-task corpora before training begins.

The only intended difference between corpora is the supervision signal.

## Frozen input boundary

Step 2 consumes the Step 1 split manifest:

```text
artifacts/transfer/splits/manifest.json
```

It verifies that the supplied episode JSONL matches the frozen SHA-256 stored by the split manifest:

```text
df4e40330ad6d3f6d4977e1630e2e54e3cfc06b01277d1aa98b7994e8c63e5ab
```

If the episode artifact changes, corpus construction fails.

## Reproduction command

```powershell
.\scripts\50-build-training-corpora.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -SplitManifestPath artifacts\transfer\splits\manifest.json `
  -OutputDirectory artifacts\transfer\corpora `
  -Seed 17
```

## Output artifacts

```text
artifacts/transfer/corpora/
├── corpus-manifest.json
├── corpus-summary.md
├── authentic_preference/
├── language_modeling_control/
├── pair_exposure_control/
├── temporal_direction_control/
├── random_label_control/
└── shuffled_preference_control/
```

Each corpus directory contains:

```text
fold-00/train.jsonl
fold-00/validation.jsonl
fold-00/test.jsonl
...
fold-09/train.jsonl
fold-09/validation.jsonl
fold-09/test.jsonl
```

The test files are emitted for audit and optional source-task reporting. Later source-task training must not train on test partitions.

## The six corpora

| Corpus | Supervision | Purpose |
|---|---|---|
| `authentic_preference` | selected candidate index | The real preference objective. |
| `language_modeling_control` | none | Controls for exposure to the same NewsEdits text. |
| `pair_exposure_control` | none | Controls for seeing the same revision pairs without a selection label. |
| `temporal_direction_control` | newer candidate index | Tests the “newness detector” explanation. |
| `random_label_control` | deterministic random candidate index | Controls for extra optimization with meaningless labels. |
| `shuffled_preference_control` | partition-shuffled selected labels | Preserves label prevalence while breaking the episode-specific preference link. |

All six corpora use the same serialized input fields:

```text
CONTEXT_BEFORE
CANDIDATE_A
CANDIDATE_B
CONTEXT_AFTER
```

Future labels are not written into corpus JSONL records. They appear only in the corpus manifest summary so the builder can audit partition balance.

## Pass criteria

Step 2 passes only if:

```text
the episode SHA-256 matches the frozen Step 1 manifest
every corpus has the same record count in every fold partition
every corpus has the same whitespace-token input budget in every fold partition
all six expected corpora are present
test partitions are marked not-for-source-training
future labels are redacted from corpus records
```

## What this step proves

A passing result supports this limited claim:

> The representation experiment has deterministic source-task corpora in which authentic preference learning and its controls consume the same row population, fold boundary, serialized input text and partition-level token budget.

It does not prove preference-to-future transfer.

It prevents a later positive result from being explained by the authentic model alone receiving more rows, different articles, different inputs or different split boundaries.

## What would fail or revise the step

The step must be revised if:

- the input episode hash differs from Step 1;
- a corpus silently drops rows;
- a control sees different input text;
- a test partition is marked trainable;
- future labels leak into corpus JSONL records;
- shuffled labels are not deterministic;
- shuffled labels do not preserve partition-level label prevalence.

## Results

Run the command, then copy the generated table and gate results from:

```text
artifacts/transfer/corpora/corpus-summary.md
```

into this section.

### Result status

```text
IMPLEMENTED — AWAITING LOCAL RUN
```

### Generated headline values

```text
Total episodes:                 PENDING
Total article lineages:         PENDING
Outer folds:                    10
Corpora:                        6
Record-count gates:             PENDING
Input-token gates:              PENDING
Future-label redaction gate:    PENDING
```

## Next step

After Step 2 passes locally, Step 3 trains the authentic and control encoders from these exact corpora. The training code must read `corpus-manifest.json` and fail or warn if source hashes, corpus names or fold assignments differ from the frozen Step 2 outputs.
