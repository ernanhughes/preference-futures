## Step Three: Give Every Objective the Same Opportunity to Change the Encoder

Step Two made the source corpora comparable. It did not make their training comparable.

Equal record counts were not enough. The independent temporal corpus was roughly seven percent longer than the authentic preference corpus. A trainer that padded only to each batch’s longest sentence could therefore give temporal direction more encoder token exposure even though both files contained the same number of rows.

Step Three removes that freedom before training begins.

Every trained regime now starts from one frozen base snapshot and receives:

```text
one tokenizer
maximum length 256
fixed max-length padding
batch size 16
600 optimizer updates
AdamW at 0.00002
60 warmup updates
FP32 precision
one deterministic fold seed
one final-checkpoint rule
no early stopping
```

That gives each fold-and-regime job exactly:

```text
600 × 16 × 256
= 2,457,600 padded encoder token positions
```

Across ten folds and six trained regimes, the confirmatory experiment contains sixty jobs and 147,456,000 padded encoder token positions.

The untouched base snapshot remains the seventh, generic arm.

### Freeze the model before looking at results

Run:

```powershell
.\scripts\00-setup.ps1 -Training

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

The human-readable model revision is resolved to an immutable commit before the model is downloaded. The repository saves the encoder and tokenizer locally, hashes the complete snapshot, reruns the Step Two verifier and freezes every corpus-file identity into one training contract.

That contract cannot select a different number of updates for a harder task. It cannot stop a weak control early. It cannot keep the best validation checkpoint for authentic preference while forcing random labels to use the last one.

Every job saves update 600.

Source validation is diagnostic only.

### Keep the future outside the trainer

The episode artifact contains the later outcome because the future probe will eventually need it.

The source trainer never receives that object directly.

Before the first batch exists, the runtime creates a new allow-listed episode containing only:

```text
episode ID
article lineage
candidate A
candidate B
context before
context after
```

The future-revision label, V2 sentence and V2 identifiers are discarded.

The source-task files themselves contain no future fields, and their frozen hashes are rechecked before each job.

This is stronger than asking the model code not to use the future. The future is absent from the object the model code can see.

### Preserve the exact Step Two input

The trainer reconstructs the exact serialized source view:

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

That detail matters. The language-adaptation control uses word-mask positions frozen in Step Two. Changing the section labels or punctuation in Step Three would silently move those masks to different words.

The code initially made precisely that mistake during implementation. The contract tests caught it before the first model run, and the serializer was changed to reproduce Step Two literally.

### Six objectives, one encoder budget

Five regimes use the same binary sequence-classification architecture:

```text
authentic preference
shuffled preference
random labels
pair exposure
temporal direction
```

Language adaptation uses deterministic whole-word masked-language modelling over the authentic pair-and-context view.

The comparison is therefore matched where the hypothesis lives: in the encoder’s input positions and update opportunities.

It is not exact total-FLOP equivalence. The masked-language-model head is larger than a binary classifier. We record that limitation rather than calling unlike heads computationally identical.

The defensible claim is:

> Every regime receives the same opportunity to alter the shared encoder under one fixed padded-token and optimizer-update budget.

### Smoke before confirmation

The first executable test is deliberately non-confirmatory:

```powershell
.\scripts\61-step3-smoke.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -Device auto `
  -SmokeSteps 2
```

This runs all six objectives on fold zero for two updates and verifies that:

```text
the base snapshot loads into every task architecture
all objectives produce finite loss
whole-word masking works
all six artifacts can be reopened and rehashed
all six jobs receive the same two-update padded-token budget
```

Smoke artifacts are written under `smoke-runs` and marked `non_confirmatory: true`. They can never satisfy the confirmatory verifier.

After smoke passes, one complete fold should be trained before launching all sixty jobs:

```powershell
.\scripts\62-train-fixed-budget-representations.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -Folds 0 `
  -Regimes all `
  -Device auto `
  -VerifyWhenComplete
```

The full run is resumable. A completed job is skipped only when its contract and persisted model hashes still match.

### Result

```text
Status:                         AWAITING LOCAL SMOKE RUN
Base model:                     distilbert/distilbert-base-uncased
Resolved immutable revision:    PENDING
Base snapshot SHA-256:          PENDING
Training contract SHA-256:      PENDING
Trained regimes:                6
Outer folds:                    10
Confirmatory jobs:              60
Updates per job:                600
Batch size:                     16
Maximum padded length:          256
Padded positions per job:       2,457,600
Six-regime smoke verification:  PENDING
```

A passing Step Three result will establish that the six source-task encoders were trained from one base representation under the same padded encoder budget and final-checkpoint rule, with the future unavailable to the source runtime.

It will not establish transfer.

It creates the representations that make the transfer question executable.
