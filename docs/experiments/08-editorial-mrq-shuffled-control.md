# Step 8.7 — Shuffled-Preference MR.Q Control

## Question

Does authentic editorial-preference supervision shape the compact MR.Q state in a way that improves future-revision prediction, compared with the same architecture trained on shuffled preference labels?

This is the final mechanistic check. Step 8.6 already established that MR.Q does not beat a properly regularised full generic representation.

## Null models

Five deterministic shuffled-label replicas are trained for every outer fold.

For each replica and fold:

- the frozen MPNet embeddings are unchanged;
- the EditorialMRQ architecture and optimizer settings are unchanged;
- selected-index labels are permuted independently within train, validation and test partitions;
- exact class counts are preserved within each partition;
- checkpoint selection uses shuffled validation log loss;
- future labels are not read until the source model is frozen.

## Downstream representations

Each shuffled source model produces the same two states as authentic MR.Q:

- `shuffled_mrq_blind`: 129-dimensional order-invariant state;
- `shuffled_mrq_choice_aware`: 193-dimensional state oriented using the authentic historical selected/rejected choice.

The future probes use the exact Step 8.4 procedure.

## Frozen decision rule

Authentic preference specificity is supported only when:

1. authentic choice-aware MR.Q beats the mean shuffled choice-aware control with a lineage-bootstrap interval entirely below zero;
2. authentic blind MR.Q beats the mean shuffled blind control with a lineage-bootstrap interval entirely below zero;
3. authentic MR.Q has a negative point estimate against at least four of five shuffled replicas in both arms.

Passing this step would show that authentic labels shape the compact MR.Q representation. It would not reverse Step 8.6 or establish superiority to the fully regularised generic baseline.

## Commands

```powershell
.\scripts\110-prepare-editorial-mrq-shuffled-control.ps1

.\scripts\111-train-editorial-mrq-shuffled-source.ps1 `
  -Replicas "all" `
  -Folds "all" `
  -Device cuda

.\scripts\112-run-editorial-mrq-shuffled-transfer.ps1 `
  -Replicas "all" `
  -Folds "all" `
  -Arms "all" `
  -Device cuda

.\scripts\113-aggregate-editorial-mrq-shuffled-control.ps1
```

The final report is written to:

```text
artifacts\step8\editorial-mrq\future-transfer\shuffled-mrq-control\aggregate.md
```
