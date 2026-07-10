# Step 8.8 — Combined XGBoost interaction check

Step 8.8 is the final exploratory nonlinear check after Step 8.7 failed to establish
preference-specific transfer in the compact MR.Q state.

## Why the sample is adequate

The experiment contains 12,056 unique episodes grouped into 3,386 article lineages.
The five shuffled controls are repeated views of those same episodes and are never counted
as additional independent observations.

Each outer fold trains on eight lineage buckets, selects tree count on one validation bucket,
and opens the remaining test bucket once. The model is deliberately shallow and strongly
regularised because the combined representation is high-dimensional.

## Feature arms

- `xgb_generic_all`: generic unoriented plus generic choice-aware geometry (5,376 dims).
- `xgb_authentic_mrq_only`: authentic blind plus choice-aware MR.Q state (322 dims).
- `xgb_generic_plus_authentic_mrq`: all generic geometry plus authentic MR.Q (5,698 dims).
- five `xgb_generic_plus_shuffled_mrq_rXX` controls with identical dimensions.

Independent shuffled hidden coordinates are never averaged as features. Each shuffled replica
is fitted and evaluated as its own complete control arm.

## Frozen XGBoost model

- objective: binary logistic;
- metric: log loss;
- tree method: histogram;
- maximum depth: 2;
- learning rate: 0.03;
- minimum child weight: 20;
- row subsample: 0.8;
- column subsample per tree: 0.25;
- L2: 30;
- L1: 1;
- maximum 1,500 rounds;
- validation-only early stopping after 75 rounds.

There is no broad hyperparameter search and no class weighting. The same configuration and
fold seed are used for every arm.

## Decision rule

A nonlinear authentic-preference interaction is supported only when:

1. generic plus authentic MR.Q beats generic-only with a paired lineage-bootstrap interval
   entirely below zero;
2. generic plus authentic MR.Q beats the mean of the five generic-plus-shuffled controls with
   an interval entirely below zero;
3. the authentic augmentation has a favourable point estimate against at least four of five
   individual shuffled augmentations.

## Commands

```powershell
.\scripts\114-prepare-editorial-mrq-xgboost.ps1

.\scripts\115-run-editorial-mrq-xgboost.ps1 `
  -Folds "all" `
  -Arms "all" `
  -Device cuda

.\scripts\116-aggregate-editorial-mrq-xgboost.ps1
```

Read the final report:

```powershell
Get-Content `
  artifacts\step8\editorial-mrq\future-transfer\xgboost-combined\aggregate.md `
  -Raw
```
