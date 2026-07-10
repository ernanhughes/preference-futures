# Step 8.5 — Editorial MR.Q specificity audit

## Motivation

Step 8.4 passed its frozen primary rule, but the MR.Q state is substantially lower-dimensional than the raw generic controls. The observed log-loss improvement could therefore reflect compression, regularisation, or calibration rather than information specifically learned from authentic editorial preference.

## Frozen diagnostic sequence

### 8.5A — representation and regularisation audit

Report for every Step 8.4 arm and fold:

- representation dimension;
- selected L2 value;
- whether selection reached the top of the frozen L2 grid;
- train, validation and test log loss;
- calibration intercept and slope on held-out predictions;
- expected calibration error using ten fixed probability bins.

### 8.5B — dimension-matched generic controls

For each fold, fit PCA on the training partition only and project the generic controls to the exact MR.Q dimensions:

- `generic_unoriented_pca`: match `mrq_blind`;
- `generic_choice_aware_pca`: match `mrq_choice_aware`.

The validation and test partitions are transformed using the training-fitted PCA. Future labels are never used in PCA fitting.

Use the same future probe architecture, train-only standardisation, L2 grid, validation selection and one-shot test evaluation as Step 8.4.

### 8.5C — extended-regularisation diagnostic

Rerun the four original and two PCA controls with a validation-selected diagnostic L2 grid extending beyond the frozen Step 8.4 maximum. This is explicitly post-result and exploratory.

## Decision

The Step 8.4 transfer result becomes compression-resistant only when authentic MR.Q beats the dimension-matched generic PCA control in pooled test log loss with a paired lineage-bootstrap 95% interval entirely below zero.

If it does, the next specificity test is an identically shaped MR.Q trained on shuffled preference labels. If it does not, Step 8.4 is interpreted as a useful learned compression/calibration result rather than evidence that preference-specific structure transfers.
