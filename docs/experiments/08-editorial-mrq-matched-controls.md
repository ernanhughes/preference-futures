# Step 8.6 — Dimension- and regularisation-matched controls

Step 8.4 passed its frozen future-transfer rule, but Step 8.5 found two substantial confounds:

- the generic controls had 16–18× more dimensions than the MR.Q states;
- every raw generic fold selected the maximum available L2 value and showed a much larger train-to-test gap.

Step 8.6 freezes two stronger controls before their outcomes are inspected.

## Controls

### Train-only PCA

The generic representations are reduced separately inside each outer fold:

- `generic_unoriented` → 129 dimensions, matching `mrq_blind`;
- `generic_choice_aware` → 193 dimensions, matching `mrq_choice_aware`.

PCA is fitted on training rows only using `torch.pca_lowrank`. Validation and test rows are projected using the frozen training mean and components. Future labels are not used by PCA.

### Extended regularisation

The unreduced generic controls are refit using the frozen L2 grid:

```text
1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
```

Selection remains validation-log-loss only. The probe architecture, preprocessing, folds and one-time test evaluation remain identical to Step 8.4.

## Decision rule

Compression and regularisation specificity is supported only when `mrq_choice_aware` beats both:

1. `pca_generic_choice_aware`; and
2. `extended_generic_choice_aware`.

For both comparisons, the pooled treatment-minus-control log-loss estimate must be negative and the paired lineage-bootstrap 95% upper endpoint must be below zero.

Passing Step 8.6 does not yet establish authentic preference-label specificity. The remaining control is an identically shaped MR.Q trained on shuffled preference labels.

## Commands

```powershell
.\scripts\107-prepare-editorial-mrq-matched-controls.ps1
.\scripts\108-run-editorial-mrq-matched-controls.ps1 -Folds "all" -Arms "all" -Device cuda
.\scripts\109-aggregate-editorial-mrq-matched-controls.ps1
```

Read the result with:

```powershell
Get-Content `
  artifacts\step8\editorial-mrq\future-transfer\matched-controls\aggregate.md `
  -Raw
```
