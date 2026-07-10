# Step 8.4 — Editorial MR.Q Future Transfer

## Entry gate

The pooled Step 8 source-task result must pass before this experiment is frozen:

- MR.Q pooled Wilson accuracy lower bound above 0.50;
- MR.Q pooled log loss below `log(2)`;
- MR.Q pooled log loss below the antisymmetric linear control;
- exact candidate-order symmetry.

The observed source result passed this gate over 12,056 out-of-fold episodes.

## Question

Does the learned MR.Q editorial-decision state improve prediction of whether the
retained sentence is revised again, beyond the same frozen semantic geometry and
the same historical selected/rejected orientation?

## Arms

| Arm | Historical choice exposed | Representation |
|---|---|---|
| `generic_unoriented` | No | context, candidate mean, absolute candidate difference |
| `generic_choice_aware` | Yes | context, selected, rejected, selected-minus-rejected |
| `mrq_blind` | No | symmetric MR.Q hidden-state and absolute value-margin features |
| `mrq_choice_aware` | Yes | selected/rejected MR.Q hidden states and signed value margin |

## Primary estimand

`mrq_choice_aware - generic_choice_aware` pooled future test log loss.

A negative value favours MR.Q. Both arms receive the authentic historical choice,
so the comparison cannot be won merely by exposing which candidate was retained.

## Secondary estimand

`mrq_blind - generic_unoriented` pooled future test log loss.

This asks whether the learned MR.Q state transfers without directly exposing the
historical choice to the downstream probe.

## Probe contract

Every arm uses the Step 6 probe procedure:

- train-only featurewise standardisation;
- one linear sigmoid probe with bias;
- identical L2 grid;
- validation-log-loss selection;
- one test evaluation after selection;
- the original lineage-grouped ten-fold partition.

## Uncertainty and decision rule

The primary loss difference receives a paired article-lineage bootstrap with seed
17 and 10,000 replicates. Future transfer is supported only when:

1. the primary point estimate is below zero; and
2. the two-sided 95% bootstrap interval lies entirely below zero.

## Commands

```powershell
.\scripts\103-prepare-editorial-mrq-transfer.ps1
.\scripts\104-run-editorial-mrq-transfer.ps1 -Folds "0"
.\scripts\104-run-editorial-mrq-transfer.ps1 -Folds "1-9"
.\scripts\105-aggregate-editorial-mrq-transfer.ps1
```

The source-task artifacts and frozen Steps 1–7 outputs are never mutated.
