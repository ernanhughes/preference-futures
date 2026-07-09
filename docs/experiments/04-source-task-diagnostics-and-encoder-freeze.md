# Step 4 — Diagnose Source Tasks and Freeze the Encoder Set

## Question

Did each Step 3 objective produce a mechanically valid encoder, and what source-task behaviour did it show before any future-outcome probe is trained?

Step 4 is diagnostic. It does not train another model, choose a new checkpoint or inspect future labels.

## Inputs

Step 4 consumes the frozen Step 3 directory:

```text
artifacts/transfer/training/contract.json
artifacts/transfer/training/training-verification-confirmatory.json
artifacts/transfer/training/runs/fold-00/...
...
artifacts/transfer/training/runs/fold-09/...
```

The full confirmatory verifier must already have passed all 60 jobs.

## Frozen distinction

Step 4 records three different properties:

```text
artifact_valid
source_task_learned
eligible_for_downstream
```

They are deliberately not aliases.

A preregistered arm remains eligible for downstream representation extraction when its encoder is mechanically valid, even if its source-task head is null-like. Source-task failure is evidence about what the source objective learned; it is not permission to remove a disappointing arm after observing the result.

Encoders are excluded only for mechanical failures such as:

```text
missing or changed encoder artifact
invalid or incomplete confirmatory run
non-finite validation metrics
```

The fixed candidate remains update 600. Step 4 cannot select an earlier checkpoint or trigger post-result retraining.

## Source-task diagnostics

For each binary task, Step 4 reconstructs the validation class prior from the frozen Step 2 JSONL and records:

```text
records
correct predictions
accuracy
95% Wilson interval
majority-class prior accuracy
class-prior log loss
observed mean loss
```

A binary task is labelled `learned_above_prior` only when:

```text
the lower 95% Wilson bound exceeds the majority-class prior
and
observed log loss is below the class-prior log loss
```

Otherwise it is labelled `null_like`, unless the full interval is below the prior.

Language adaptation is evaluated separately. Step 4 checks finite metrics, loss reduction across the committed training trajectory and mask fallbacks. MLM loss is never compared directly with binary classification loss.

## Hash audit

Step 4 independently re-hashes every encoder directory and requires:

```text
60 trained encoders
60 unique trained encoder hashes
no trained encoder identical to the untouched base encoder
```

The untouched base encoder is then repeated as the generic seventh arm for each fold.

## Command

```powershell
.\scripts\70-freeze-source-task-encoders.ps1 `
  -TrainingDirectory artifacts\transfer\training `
  -OutputDirectory artifacts\transfer\encoder-selection
```

## Outputs

```text
artifacts/transfer/encoder-selection/
├── source-task-summary.json
├── source-task-summary.md
├── accepted-encoders.json
├── encoder-hash-audit.json
└── trajectory-summary.json
```

`accepted-encoders.json` is the only encoder inventory Step 5 may consume.

For ten folds and seven arms, a complete manifest contains:

```text
10 × 7 = 70 entries
```

## Interpretation rule

A null-like authentic-preference source classifier weakens the claim that the editor's immediate choice was directly learned under the frozen contract. It does not by itself answer whether those updates changed the encoder in a way that helps a later future probe.

That stronger transfer question remains Step 6.

## Supported claim after a passing run

> All mechanically valid Step 3 encoders were independently re-hashed, diagnosed against task-appropriate source baselines and frozen in a seven-arm, ten-fold manifest without using source-task success to select favourable downstream arms.

## Results

```text
IMPLEMENTED — AWAITING LOCAL STEP 4 ARTIFACT GENERATION
```
