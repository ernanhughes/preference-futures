# Step 5 — Extract the Frozen Representations

## Question

What fixed representation does each Step 4 encoder produce for the exact same preference episode before any future-outcome probe is trained?

Step 5 performs inference only. It does not train, calibrate or select a future probe.

## Inputs

Step 5 consumes only frozen artifacts from earlier steps:

```text
artifacts/transfer/encoder-selection/accepted-encoders.json
artifacts/transfer/training/contract.json
artifacts/transfer/splits/manifest.json
artifacts/newsedits/viability-5000/episodes.jsonl
```

The Step 4 manifest supplies ten folds and seven arms per fold:

```text
generic
language_adaptation
pair_exposure
temporal_direction
random_label
shuffled_preference
authentic_preference
```

All 70 entries must remain mechanically valid and eligible.

## Frozen episode view

Every arm receives the same canonical pair-and-context serialization:

```text
[CONTEXT_BEFORE]
...
[CANDIDATE_A]
...
[CANDIDATE_B]
...
[CONTEXT_AFTER]
...
```

Candidate order remains the deterministic Step 1 presentation order.

The source loader projects the future-bearing episode record to:

```text
episode_id
lineage_id
candidate_a
candidate_b
context_before
context_after
```

The encoder never receives:

```text
selected_index
future_revised
V2 text
V2 identifiers
numeric-control flags
```

The representation row metadata contains only:

```text
row_index
episode_id
lineage_id
input_sha256
```

Step 6 joins the future label from the frozen episode source after representation extraction.

## Frozen representation rule

The extraction contract uses:

| Setting | Value |
|---|---|
| Model object | encoder only through `AutoModel` |
| Source-task head | not loaded |
| Tokenizer | frozen Step 3 tokenizer |
| Maximum length | 256 |
| Padding | fixed `max_length` |
| Truncation | enabled |
| Model mode | evaluation |
| Gradient tracking | disabled |
| Pooling | final-layer first-token state |
| Pooling token | `[CLS]` |
| Output dtype | float32 |
| Default batch size | 32 |

The final-layer first-token state is frozen because DistilBERT sequence classification consumes that state before its task-specific pre-classifier and output head.

Step 5 does not extract several pooling variants and choose one later. A mean-pooled or multi-layer representation would be a separate experiment.

## Partition contract

For outer fold `i`:

```text
test       = lineage bucket i
validation = lineage bucket (i + 1) mod 10
train      = all remaining lineage buckets
```

Rows are sorted by `episode_id` inside each partition. Every arm in a fold must have exactly the same row order and input hashes.

## Prepare before extraction

```powershell
.\scripts\80-prepare-frozen-representations.ps1 `
  -SelectionManifest artifacts\transfer\encoder-selection\accepted-encoders.json `
  -TrainingDirectory artifacts\transfer\training `
  -OutputDirectory artifacts\transfer\representations `
  -BatchSize 32
```

Preparation re-hashes every selected encoder and freezes:

```text
70 extraction jobs
210 partition matrices
one tokenizer
one input serialization
one pooling rule
one output dtype
```

Once any representation is observed, do not regenerate the contract with a different pooling rule, maximum length or batch interpretation.

## Recommended pilot

Run the generic and authentic arms on fold 0 first:

```powershell
.\scripts\81-extract-frozen-representations.ps1 `
  -RepresentationDirectory artifacts\transfer\representations `
  -Folds 0 `
  -Arms generic,authentic_preference `
  -Device cuda `
  -VerifyWhenComplete
```

This produces six matrices and checks the full persistence path without changing the confirmatory contract.

## Run or resume all jobs

```powershell
.\scripts\81-extract-frozen-representations.ps1 `
  -RepresentationDirectory artifacts\transfer\representations `
  -Folds all `
  -Arms all `
  -Device cuda
```

Completed jobs are skipped only when their contract, encoder and output hashes still match. Do not use `-Force` unless replacing an invalid artifact deliberately.

The generic encoder is shared across folds. Its full episode matrix is computed once per invocation and partitioned according to each fold. The 60 trained encoders remain fold-specific.

## Per-job artifacts

Each fold and arm writes:

```text
artifacts/transfer/representations/runs/fold-00/authentic_preference/
├── train.safetensors
├── train.rows.jsonl
├── validation.safetensors
├── validation.rows.jsonl
├── test.safetensors
├── test.rows.jsonl
└── run.json
```

Each Safetensors file contains one tensor:

```text
representations: [rows, hidden_size], float32
```

## Verify all persisted matrices

```powershell
.\scripts\82-verify-frozen-representations.ps1 `
  -RepresentationDirectory artifacts\transfer\representations `
  -Folds all `
  -Arms all
```

The verifier requires:

```text
70/70 run reports
210/210 partition matrices
all artifact hashes valid
one hidden size
one device type
one runtime environment
finite float32 vectors
exact frozen partition membership
identical row order across arms
no selected or future labels in row metadata
all episodes tested exactly once across outer folds
```

It writes:

```text
representation-verification.json
representation-verification.md
```

## What a passing Step 5 proves

> The seven frozen encoder arms were applied to one identical label-free episode view under one tokenizer, maximum length, pooling rule and output dtype, producing verified train, validation and test matrices aligned to the frozen article-lineage partitions.

It does not prove that one representation predicts the future better than another. Step 6 trains identical future probes.

## Results

```text
IMPLEMENTED — AWAITING LOCAL CONTRACT, PILOT AND FULL EXTRACTION
```
