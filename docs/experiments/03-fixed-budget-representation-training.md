# Step 3 — Train the Six Representations Under One Fixed Budget

## Question

Can all six additional-training regimes be given the same opportunity to change the encoder before we ask whether any frozen representation predicts the future?

Step 2 froze the source-task data. Step 3 freezes and executes the optimisation contract.

It still does **not** test future prediction. It produces the six trained encoders that later steps will freeze, inspect and probe.

## Inputs

Step 3 consumes only the verified Step 2 artifacts and the frozen episode source:

```text
artifacts/transfer/corpora/manifest.json
artifacts/transfer/corpora/temporal-pairs.jsonl
artifacts/transfer/corpora/fold-00/...
...
artifacts/transfer/corpora/fold-09/...
artifacts/newsedits/viability-5000/episodes.jsonl
```

Before preparing a model, the trainer reruns the persisted Step 2 verifier and checks the episode and temporal-pair SHA-256 identities recorded in the Step 2 manifest.

A changed source artifact aborts preparation.

## The seven arms

Six arms receive source-task training:

```text
language_adaptation
pair_exposure
temporal_direction
random_label
shuffled_preference
authentic_preference
```

The seventh arm is the untouched generic base encoder saved during preparation. It receives no source-task updates.

Across ten outer folds, Step 3 therefore runs:

```text
10 folds × 6 trained regimes = 60 training jobs
```

## Frozen default model

The default base model is:

```text
distilbert/distilbert-base-uncased
```

The human-readable revision defaults to `main`, but preparation resolves it to the immutable Hugging Face commit SHA before downloading anything.

Preparation saves one local snapshot:

```text
artifacts/transfer/training/base-snapshot/
├── encoder/
└── tokenizer/
```

Every fold and regime starts from the exact same encoder snapshot hash. The task head is initialised only after resetting the deterministic fold seed.

The resolved model revision, library versions, encoder class, tokenizer class and complete snapshot hash are frozen in:

```text
artifacts/transfer/training/model-source.json
artifacts/transfer/training/contract.json
```

## Frozen optimisation contract

The confirmatory defaults are:

| Setting | Value |
|---|---:|
| Precision | FP32 |
| Maximum sequence length | 256 |
| Padding | fixed `max_length` |
| Batch size | 16 |
| Gradient accumulation | 1 |
| Optimizer updates | 600 |
| Optimizer | AdamW |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Warmup updates | 60 |
| Scheduler | linear warmup, then linear decay |
| Gradient clipping | 1.0 |
| Checkpoint rule | final update 600 |
| Early stopping | forbidden |

This yields exactly:

```text
600 updates × 16 examples × 256 padded positions
= 2,457,600 padded encoder token positions per job
```

Across all 60 trained jobs:

```text
147,456,000 padded encoder token positions
```

The number of source records differs slightly by fold, so the fixed update budget is deliberately independent of corpus length. Each job consumes 9,600 example presentations. A fold with 9,643–9,646 training records is therefore approximately one deterministic pass.

## What is actually matched

Step 3 matches the encoder’s optimisation opportunity:

```text
same encoder checkpoint
same tokenizer
same maximum sequence length
same fixed padding
same batch size
same update count
same optimiser and schedule
same precision
same gradient clipping
same final-checkpoint rule
same fold seed policy
```

This removes the raw seven-percent temporal-text-length advantage observed in Step 2 from the padded encoder token budget.

It does not claim mathematically identical total FLOPs. The masked-language-model head is much larger than the binary classification heads. The correct statement is:

> The six regimes receive matched encoder input positions and matched encoder update opportunities under one fixed trainer.

Do not describe this as exact wall-clock or total-head compute equivalence.

## Future-label isolation

The original episode JSONL contains the later outcome because future probes will need it.

Step 3 does not expose that record directly to the model runtime. During source loading, it constructs a new allow-listed object containing only:

```text
episode_id
lineage_id
candidate_a
candidate_b
context_before
context_after
```

Everything else—including `future_revised`, V2 text and V2 identifiers—is discarded before a batch can be materialised.

The compact Step 2 source-task files are also rechecked against their frozen hashes before every job.

## Exact source views

Step 3 reproduces the Step 2 serialized input exactly:

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

This matters for language adaptation because Step 2 froze mask positions over whitespace words in that exact view. Changing labels, punctuation or section markers would silently move the masks to different words.

## The six objectives

### Authentic preference

Binary sequence classification:

```text
predict the editor-retained candidate index
```

### Shuffled preference

The same binary architecture and authentic input, but the target comes from a different article lineage while preserving partition-level label counts.

### Random label

The same binary architecture and authentic input, with deterministic balanced random targets.

### Pair exposure

Binary sequence classification:

```text
predict whether candidate A and candidate B came from the same revision episode
```

Negative examples substitute candidate B from a different lineage.

### Temporal direction

Binary sequence classification over evaluation-disjoint NewsEdits revisions:

```text
predict the newer candidate index
```

The earlier and later candidates are oriented so the frozen target always identifies the later sentence.

### Language adaptation

Deterministic whole-word masked-language modelling over the authentic Step 2 pair-and-context view.

Step 2’s mask indices select whitespace words. The fast tokenizer expands each selected word to all of its subword positions. A truncated example with no selected word remaining receives one deterministic fallback word, and the fallback count is recorded.

## Installation

The normal repository environment remains lightweight. Install the model-training stack explicitly:

```powershell
.\scripts\00-setup.ps1 -Training
```

This installs the optional `train` dependency group containing PyTorch, Transformers, Hugging Face Hub and Safetensors.

## Prepare and freeze the contract

Run once:

```powershell
.\scripts\60-prepare-fixed-budget-training.ps1 `
  -CorporaDirectory artifacts\transfer\corpora `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -OutputDirectory artifacts\transfer\training `
  -ModelId "distilbert/distilbert-base-uncased" `
  -ModelRevision "main" `
  -Seed 17 `
  -MaximumSequenceLength 256 `
  -BatchSize 16 `
  -UpdateSteps 600
```

The command resolves the immutable model revision, saves the base snapshot, validates Step 2 and writes:

```text
artifacts/transfer/training/
├── contract.json
├── training-plan.md
├── model-source.json
└── base-snapshot/
```

Once any confirmatory result has been observed, do not regenerate this directory with different hyperparameters. A changed contract is a new experiment.

## Six-regime smoke run

Before starting the 60 confirmatory jobs, execute two updates for all six regimes on fold 0:

```powershell
.\scripts\61-step3-smoke.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -Device auto `
  -SmokeSteps 2
```

Smoke outputs are isolated under:

```text
artifacts/transfer/training/smoke-runs/
```

They are permanently marked `non_confirmatory: true` and cannot satisfy the confirmatory verifier.

The smoke run checks that:

```text
the base encoder loads strictly into every task architecture
the tokenizer and whole-word masks work
each objective produces finite loss
the six final smoke artifacts can be reopened and hashed
the verifier sees identical update and padded-token budgets
```

## Confirmatory training

The command is resumable. Completed runs whose contract and artifact hashes still match are skipped.

A useful first confirmatory stage is one complete fold:

```powershell
.\scripts\62-train-fixed-budget-representations.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -Folds 0 `
  -Regimes all `
  -Device auto `
  -VerifyWhenComplete
```

After that passes, run all remaining jobs:

```powershell
.\scripts\62-train-fixed-budget-representations.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -Folds all `
  -Regimes all `
  -Device auto
```

Then verify the complete set:

```powershell
.\scripts\63-verify-fixed-budget-training.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -Folds all `
  -Regimes all
```

## Per-job artifacts

Each confirmatory job writes atomically to:

```text
artifacts/transfer/training/runs/fold-00/authentic_preference/
├── run.json
├── metrics.jsonl
├── task-model/
├── encoder/
└── tokenizer/
```

The `encoder/` directory is the artifact consumed by later representation extraction. The source-task head is retained separately in `task-model/` for Step 4 diagnostics.

Each `run.json` records:

```text
contract hash
fold and regime
source corpus hashes
initial encoder snapshot hash
optimizer updates completed
example presentations
padded token positions
fixed checkpoint step
validation metrics
model and tokenizer artifact hashes
Python, PyTorch and Transformers versions
device and CUDA information
```

## Persisted verification gates

The confirmatory verifier passes only when:

```text
all 60 expected runs exist
all runs use the frozen contract hash
all train and validation corpus hashes still match
all encoders start from the frozen base snapshot
every job completes exactly 600 updates
every job records exactly 2,457,600 padded token positions
every job saves checkpoint step 600
no run uses early stopping
all task-model, encoder and tokenizer hashes survive reopening
all validation metrics are finite
one device type is used for the selected comparison set
all six regimes have equal budgets within every fold
```

## Checkpoint selection

There is no source-task model selection in Step 3.

Validation is computed once after the final update and recorded for diagnosis. It cannot select an earlier checkpoint, a different update count or a favourable regime-specific stopping point.

This prevents the future experiment from receiving six differently tuned representations.

## What a passing Step 3 proves

A complete passing run supports only this statement:

> Six source-task encoders were trained from one frozen base snapshot under the same tokenizer, padded sequence budget, batch size, optimiser schedule, update count and final-checkpoint rule, with future fields unavailable to the source-task runtime.

It does not prove that the source tasks were equally learnable.

It does not prove that authentic preference training learned the editor’s decision better than its controls. Step 4 checks source-task learning and freezes the accepted encoder set.

It does not prove future transfer. That requires the later frozen-representation probe.

## What would fail or revise the step

Step 3 must be revised before future probing if:

- any source hash differs from Step 2;
- any task sees a future or V2 field;
- any regime starts from a different encoder snapshot;
- any regime receives a different update or padded-token budget;
- an earlier checkpoint is selected from source validation;
- a job uses early stopping;
- a saved artifact cannot reproduce its recorded hash;
- the language-adaptation masks no longer refer to the Step 2 serialized words;
- the full comparison mixes CPU and GPU runs without a declared sensitivity analysis;
- any run produces non-finite loss or metrics.

## Results

### Result status

```text
IMPLEMENTED — AWAITING LOCAL MODEL SNAPSHOT AND SIX-REGIME SMOKE RUN
```

### First values to record

After preparation and smoke, copy:

```text
artifacts/transfer/training/training-plan.md
artifacts/transfer/training/training-verification-smoke.md
artifacts/transfer/training/smoke-runs/fold-00/*/run.json
```

Record:

```text
resolved immutable model revision
base snapshot SHA-256
contract SHA-256
smoke device and library versions
six smoke validation losses
six smoke encoder artifact hashes
mask fallback count
all smoke verification gates
```

Do not begin interpreting source-task rankings from a two-update smoke run.

## Next step

Step 4 verifies that the six source objectives behaved as intended, checks for failed or degenerate training, and freezes the encoder artifacts that Step 5 will use for representation extraction.
